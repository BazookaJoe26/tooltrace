// Camera calibration and perspective correction in physical paper coordinates.
use crate::{
    calibration_fit::{Point, corner_order, fit_corners},
    detection::TagDetection,
};
use ab_glyph::{FontRef, PxScale};
use anyhow::{Context, Result, bail};
use image::{Rgb, RgbImage};
use imageproc::drawing::{draw_filled_circle_mut, draw_line_segment_mut, draw_text_mut};
use opencv::{
    calib3d::find_homography,
    core::{BORDER_CONSTANT, Mat, Point2d, Scalar, Size, Vector},
    imgproc::{INTER_LINEAR, warp_perspective},
    prelude::{MatTraitConst, MatTraitConstManual},
};
use tooltrace_common::PaperSize;

pub const PIXELS_PER_MM: f64 = 300.0 / 25.4;

pub struct CameraCalibration {
    pub homography_mat: Mat,
    pub output_width: u32,
    pub output_height: u32,
    pub rotated_image: Option<RgbImage>,
    pub paper_size: PaperSize,
    pub fit_max_mm: f64,
    pub fit_report: String,
}

/// Fit all sixteen black-square corners, grouped by the four printed tag IDs.
/// Centers initialize corner assignment only; they are not fitting observations.
pub fn calculate_calibration(
    detections: &[TagDetection],
    tag_size_mm: f64,
    input_image: &RgbImage,
    debug_prefix: Option<&str>,
) -> Result<CameraCalibration> {
    let ids: Vec<u32> = detections.iter().map(|d| d.id).collect();
    let paper_size = tooltrace_common::AprilTagConfig::detect_paper_size(&ids)
        .context("Need all four paper corner tags (IDs 0-3, 4-7 or 8-11)")?;
    let (width, height) = paper_size.dimensions_mm();
    if !tag_size_mm.is_finite()
        || tag_size_mm <= 0.0
        || 2.0 * tag_size_mm + 30.0 >= width.min(height)
    {
        bail!("Invalid tag size for the detected paper");
    }
    let base_id = match paper_size {
        PaperSize::A4 => 0,
        PaperSize::Letter => 4,
        PaperSize::A3 => 8,
    };
    let tags: Vec<&TagDetection> = (base_id..base_id + 4)
        .map(|id| {
            detections
                .iter()
                .filter(|d| d.id == id)
                .max_by(|a, b| a.decision_margin.total_cmp(&b.decision_margin))
                .with_context(|| format!("Missing corner tag ID {id}"))
        })
        .collect::<Result<_>>()?;
    let offset = 15.0 + tag_size_mm / 2.0;
    let centers_mm = [
        (offset, offset),
        (width - offset, offset),
        (width - offset, height - offset),
        (offset, height - offset),
    ];
    let source_centers: Vec<Point> = tags.iter().map(|d| d.center).collect();
    if !source_centers
        .iter()
        .all(|p| p.0.is_finite() && p.1.is_finite())
    {
        bail!("Non-finite marker center");
    }
    let target_centers: Vec<Point> = centers_mm
        .iter()
        .map(|p| (p.0 * PIXELS_PER_MM, p.1 * PIXELS_PER_MM))
        .collect();
    let initial = initial_homography(&source_centers, &target_centers)?;
    let mut source = Vec::with_capacity(16);
    let mut target = Vec::with_capacity(16);
    let half = tag_size_mm / 2.0;
    for (tag, &(x, y)) in tags.iter().zip(&centers_mm) {
        let expected = [
            (x - half, y - half),
            (x + half, y - half),
            (x + half, y + half),
            (x - half, y + half),
        ]
        .map(|p| (p.0 * PIXELS_PER_MM, p.1 * PIXELS_PER_MM));
        let ordered = corner_order(&tag.corners, &expected, &initial)?;
        let area = (0..4)
            .map(|i| {
                let (a, b) = (ordered[i], ordered[(i + 1) % 4]);
                a.0 * b.1 - a.1 * b.0
            })
            .sum::<f64>()
            .abs()
            / 2.0;
        if area < 1.0 {
            bail!("Degenerate corners for tag {}", tag.id);
        }
        source.extend(ordered);
        target.extend(expected);
    }
    let fit = fit_corners(&source, &target, &initial, 0.35 * PIXELS_PER_MM)?;
    let errors_mm: Vec<f64> = fit.residuals_px.iter().map(|e| e / PIXELS_PER_MM).collect();
    let fit_rms_mm = (errors_mm.iter().map(|e| e * e).sum::<f64>() / 16.0).sqrt();
    let fit_max_mm = errors_mm.iter().copied().fold(0.0_f64, f64::max);
    let mut fit_report = format!(
        "Detected paper size: {paper_size}\nCalibration: all 16 marker corners, robust fit (Huber threshold 0.35 mm)\n  Corner residuals: RMS {fit_rms_mm:.3} mm, maximum {fit_max_mm:.3} mm\n  Reduced weight: {} of 16 corners\n",
        fit.downweighted
    );
    for (i, tag) in tags.iter().enumerate() {
        let e = &errors_mm[i * 4..i * 4 + 4];
        fit_report.push_str(&format!(
            "  Tag {} corner residuals (TL/TR/BR/BL): {:.3}, {:.3}, {:.3}, {:.3} mm\n",
            tag.id, e[0], e[1], e[2], e[3]
        ));
    }
    fit_report
        .push_str("  These are fitting residuals; the printed rulers are checked separately.\n");
    println!("{fit_report}");
    let calibration = CameraCalibration {
        homography_mat: fit.homography,
        output_width: (width * PIXELS_PER_MM).ceil() as u32,
        output_height: (height * PIXELS_PER_MM).ceil() as u32,
        rotated_image: Some(input_image.clone()),
        paper_size,
        fit_max_mm,
        fit_report,
    };
    if let Some(prefix) = debug_prefix {
        let font = FontRef::try_from_slice(include_bytes!("../../fonts/NotoSans-Regular.ttf"))?;
        let mut annotated = input_image.clone();
        for i in 0..16 {
            let p = source[i];
            let next = source[i / 4 * 4 + (i + 1) % 4];
            let color = if errors_mm[i] > 0.35 {
                Rgb([255, 70, 0])
            } else {
                Rgb([0, 255, 255])
            };
            draw_line_segment_mut(
                &mut annotated,
                (p.0 as f32, p.1 as f32),
                (next.0 as f32, next.1 as f32),
                color,
            );
            draw_filled_circle_mut(&mut annotated, (p.0 as i32, p.1 as i32), 5, color);
            draw_text_mut(
                &mut annotated,
                color,
                p.0 as i32 + 8,
                p.1 as i32 + 8,
                PxScale::from(18.0),
                &font,
                &format!("{:.2}mm", errors_mm[i]),
            );
        }
        annotated.save(format!("{prefix}_cropped_annotated.jpg"))?;
        let mut flattened = apply_perspective_correction(&annotated, &calibration)?;
        for (x, y) in target {
            draw_line_segment_mut(
                &mut flattened,
                ((x - 8.0) as f32, y as f32),
                ((x + 8.0) as f32, y as f32),
                Rgb([255, 0, 255]),
            );
            draw_line_segment_mut(
                &mut flattened,
                (x as f32, (y - 8.0) as f32),
                (x as f32, (y + 8.0) as f32),
                Rgb([255, 0, 255]),
            );
        }
        flattened.save(format!("{prefix}_flattened_annotated.jpg"))?;
    }
    Ok(calibration)
}

fn initial_homography(source: &[Point], target: &[Point]) -> Result<Mat> {
    let source: Vector<Point2d> = source.iter().map(|p| Point2d::new(p.0, p.1)).collect();
    let target: Vector<Point2d> = target.iter().map(|p| Point2d::new(p.0, p.1)).collect();
    let h = find_homography(&source, &target, &mut Mat::default(), 0, 3.0)?;
    if h.empty() {
        bail!("Degenerate marker center layout");
    }
    Ok(h)
}

/// Apply perspective correction to warp image to flat paper coordinates using OpenCV
pub fn apply_perspective_correction(
    source_img: &RgbImage,
    calibration: &CameraCalibration,
) -> Result<RgbImage> {
    let (src_width, src_height) = source_img.dimensions();

    println!(
        "Warping image from {}x{} to {}x{} using OpenCV",
        src_width, src_height, calibration.output_width, calibration.output_height
    );

    // Convert RgbImage to OpenCV Mat
    let src_mat_raw =
        Mat::from_slice(source_img.as_raw()).context("Failed to create Mat from image")?;
    let src_mat = src_mat_raw
        .reshape_nd(3, &[src_height as i32, src_width as i32])
        .context("Failed to reshape Mat")?;

    // Create output Mat
    let mut dst_mat = Mat::default();
    let dst_size = Size::new(
        calibration.output_width as i32,
        calibration.output_height as i32,
    );

    // Apply perspective warp
    println!("Applying perspective warp with OpenCV...");
    warp_perspective(
        &src_mat,
        &mut dst_mat,
        &calibration.homography_mat,
        dst_size,
        INTER_LINEAR,
        BORDER_CONSTANT,
        Scalar::default(),
    )
    .context("Failed to warp perspective")?;

    // Convert back to RgbImage
    let dst_data = dst_mat.data_bytes().context("Failed to get output data")?;
    let output = RgbImage::from_raw(
        calibration.output_width,
        calibration.output_height,
        dst_data.to_vec(),
    )
    .context("Failed to create output image")?;

    println!("Warping complete!");

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_distances_survive_perspective_and_rotation_without_raster_rounding() {
        let ppm = 300.0 / 25.4;
        for (paper, base_id) in [
            (PaperSize::A4, 0),
            (PaperSize::Letter, 4),
            (PaperSize::A3, 8),
        ] {
            let (width, height) = paper.dimensions_mm();
            for rotation in 0..5 {
                let photograph = |x: f64, y: f64| {
                    let denominator = 1.0 + 0.0003 * x + 0.0002 * y;
                    let (x, y) = (
                        (8.0 * x + 0.3 * y + 100.0) / denominator,
                        (0.2 * x + 8.0 * y + 200.0) / denominator,
                    );
                    match rotation {
                        0 => (x, y),
                        1 => (y, 4000.0 - x),
                        2 => (4000.0 - x, 4000.0 - y),
                        3 => (4000.0 - y, x),
                        _ => {
                            let angle = 33.0_f64.to_radians();
                            (
                                2000.0 + angle.cos() * x - angle.sin() * y,
                                200.0 + angle.sin() * x + angle.cos() * y,
                            )
                        }
                    }
                };
                let positions = [
                    (35.0, 35.0),
                    (width - 35.0, 35.0),
                    (width - 35.0, height - 35.0),
                    (35.0, height - 35.0),
                ];
                let tags: Vec<TagDetection> = positions
                    .iter()
                    .enumerate()
                    .map(|(i, &(x, y))| TagDetection {
                        id: base_id + i as u32,
                        center: photograph(x, y),
                        corners: [
                            photograph(x - 20.0, y - 20.0),
                            photograph(x + 20.0, y - 20.0),
                            photograph(x + 20.0, y + 20.0),
                            photograph(x - 20.0, y + 20.0),
                        ],
                        hamming_distance: 0,
                        decision_margin: 100.0,
                    })
                    .collect();
                let calibration =
                    calculate_calibration(&tags, 40.0, &RgbImage::new(1, 1), None).unwrap();
                let mapped = |point: (f64, f64)| {
                    let (x, y) = photograph(point.0, point.1);
                    let h = |r, c| *calibration.homography_mat.at_2d::<f64>(r, c).unwrap();
                    let w = h(2, 0) * x + h(2, 1) * y + h(2, 2);
                    (
                        (h(0, 0) * x + h(0, 1) * y + h(0, 2)) / w / ppm,
                        (h(1, 0) * x + h(1, 1) * y + h(1, 2)) / w / ppm,
                    )
                };
                // These ruler points were not among the sixteen fitted corners.
                for (a, b, expected) in [
                    (
                        (65.0, height - 35.0),
                        (width - 65.0, height - 35.0),
                        width - 130.0,
                    ),
                    ((35.0, 65.0), (35.0, height - 65.0), height - 130.0),
                ] {
                    let (a, b) = (mapped(a), mapped(b));
                    let length = (b.0 - a.0).hypot(b.1 - a.1);
                    assert!(
                        (length - expected).abs() < 0.0001,
                        "{paper:?}, rotation {rotation}: {length} mm, expected {expected} mm"
                    );
                }
            }
        }
    }
}
