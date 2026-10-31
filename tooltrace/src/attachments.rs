//! Recover small FastSAM instances that overlap the selected main object.
//! Eligibility is tested against the original main mask, never recursively
//! against newly added parts. Attachments are merged after line regularization.
use anyhow::{bail, Result};
use image::{Rgb, RgbImage};
use imageproc::drawing::draw_line_segment_mut;
use opencv::{
    core::{Mat, Point, Scalar, Vector, CV_8UC1},
    imgproc,
    prelude::*,
};
use tooltrace_common::{Contour, Point2DMm};

const PPM: f64 = 300.0 / 25.4;
const MAX_PART_SIZE_MM: f64 = 20.0;
const MAX_AREA_FRACTION: f64 = 0.05;

fn polygon(contour: &Contour) -> Option<Vector<Point>> {
    if contour.points.len() < 3
        || contour
            .points
            .iter()
            .any(|p| !p.x.is_finite() || !p.y.is_finite())
    {
        return None;
    }
    Some(
        contour
            .points
            .iter()
            .map(|p| Point::new(p.x.round() as i32, p.y.round() as i32))
            .collect(),
    )
}

fn mask(contours: &[Contour], width: i32, height: i32) -> Result<Mat> {
    let mut output = Mat::new_rows_cols_with_default(height, width, CV_8UC1, Scalar::all(0.0))?;
    for contour in contours {
        if let Some(points) = polygon(contour) {
            let polygons: Vector<Vector<Point>> = Vector::from_iter([points]);
            imgproc::fill_poly(
                &mut output,
                &polygons,
                Scalar::all(255.0),
                imgproc::LINE_8,
                0,
                Point::new(0, 0),
            )?;
        }
    }
    Ok(output)
}

pub fn complete_attachments(
    primary: Vec<Contour>,
    original_primary: &[Contour],
    candidates: &[Contour],
    image: &RgbImage,
    tag_regions: &[(i32, i32, i32, i32)],
    debug_path: Option<&str>,
) -> Result<Vec<Contour>> {
    if primary.is_empty() || original_primary.is_empty() {
        return Ok(primary);
    }
    let (width, height) = (image.width() as i32, image.height() as i32);
    let reference = mask(original_primary, width, height)?;
    let reference_pixels = reference.data_bytes()?;
    let main_area = opencv::core::count_non_zero(&reference)? as f64;
    if main_area <= 0.0 {
        return Ok(primary);
    }
    let mut output = mask(&primary, width, height)?;
    let mut accepted = 0;
    for candidate in candidates {
        let Some(points) = polygon(candidate) else {
            continue;
        };
        let bbox = imgproc::bounding_rect(&points)?;
        let area = imgproc::contour_area(&points, false)?;
        if area < PPM * PPM
            || area > MAX_AREA_FRACTION * main_area
            || bbox.width.max(bbox.height) as f64 > MAX_PART_SIZE_MM * PPM
        {
            continue;
        }
        let perimeter = imgproc::arc_length(&points, true)?;
        if perimeter <= 0.0 || 4.0 * std::f64::consts::PI * area / perimeter.powi(2) < 0.12 {
            continue;
        }
        // Require the entire proposal to lie inside the rectified image.
        if bbox.x < 0 || bbox.y < 0 || bbox.x + bbox.width > width || bbox.y + bbox.height > height
        {
            continue;
        }
        let part = mask(std::slice::from_ref(candidate), width, height)?;
        let pixels = part.data_bytes()?;
        let mut overlap = 0usize;
        let mut outside = 0usize;
        let mut touches_tag = false;
        for y in bbox.y..bbox.y + bbox.height {
            for x in bbox.x..bbox.x + bbox.width {
                let index = (y * width + x) as usize;
                if pixels[index] == 0 {
                    continue;
                }
                if reference_pixels[index] != 0 {
                    overlap += 1;
                } else {
                    outside += 1;
                }
                if tag_regions
                    .iter()
                    .any(|&(tx, ty, tw, th)| x >= tx && x < tx + tw && y >= ty && y < ty + th)
                {
                    touches_tag = true;
                }
            }
        }
        let total = (overlap + outside) as f64;
        // Exclude internal details, nearby disconnected objects/printing, and
        // large alternatives to the primary mask. No gap is bridged or filled.
        if touches_tag
            || overlap < 9
            || (overlap as f64) < 0.05 * total
            || (outside as f64) < 0.15 * total
            || (outside as f64) < PPM * PPM
        {
            continue;
        }
        let mut union = Mat::default();
        opencv::core::bitwise_or(&output, &part, &mut union, &Mat::default())?;
        // Straightening can move the original contact by a few pixels. Do not
        // invent a connecting bridge when the actual masks no longer touch.
        let mut components: Vector<Vector<Point>> = Vector::new();
        imgproc::find_contours_def(
            &union,
            &mut components,
            imgproc::RETR_EXTERNAL,
            imgproc::CHAIN_APPROX_SIMPLE,
        )?;
        if components.len() != 1 {
            continue;
        }
        output = union;
        accepted += 1;
        println!(
            "  Attached FastSAM part: {:.1} x {:.1} mm, added {:.1} mm²",
            bbox.width as f64 / PPM,
            bbox.height as f64 / PPM,
            outside as f64 / PPM.powi(2)
        );
    }
    println!("Recovered {accepted} attached FastSAM part(s)");
    if accepted == 0 {
        return Ok(primary);
    }
    let mut contours: Vector<Vector<Point>> = Vector::new();
    imgproc::find_contours_def(
        &output,
        &mut contours,
        imgproc::RETR_EXTERNAL,
        imgproc::CHAIN_APPROX_SIMPLE,
    )?;
    if contours.len() != 1 {
        bail!("Attachment union unexpectedly disconnected");
    }
    let points = contours.get(0)?;
    let result = Contour {
        points: points
            .iter()
            .map(|p| Point2DMm::new(p.x as f64, p.y as f64))
            .collect(),
        closed: true,
    };
    if let Some(path) = debug_path {
        let mut overlay = image.clone();
        for i in 0..result.points.len() {
            let (a, b) = (
                result.points[i],
                result.points[(i + 1) % result.points.len()],
            );
            draw_line_segment_mut(
                &mut overlay,
                (a.x as f32, a.y as f32),
                (b.x as f32, b.y as f32),
                Rgb([0, 255, 0]),
            );
        }
        overlay.save(path)?;
    }
    Ok(vec![result])
}

#[cfg(test)]
mod tests {
    use super::*;
    fn rectangle(x: f64, y: f64, w: f64, h: f64) -> Contour {
        Contour {
            points: [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
                .map(|p| Point2DMm::new(p.0, p.1))
                .to_vec(),
            closed: true,
        }
    }
    fn circle(x: f64, y: f64, r: f64) -> Contour {
        Contour {
            points: (0..80)
                .map(|i| {
                    let a = i as f64 / 80.0 * 2.0 * std::f64::consts::PI;
                    Point2DMm::new(x + r * a.cos(), y + r * a.sin())
                })
                .collect(),
            closed: true,
        }
    }
    fn contains(contours: &[Contour], x: f32, y: f32) -> bool {
        imgproc::point_polygon_test(
            &polygon(&contours[0]).unwrap(),
            opencv::core::Point2f::new(x, y),
            false,
        )
        .unwrap()
            >= 0.0
    }

    #[test]
    fn recovers_round_parts_but_rejects_nearby_objects_background_and_chained_parts() {
        let image = RgbImage::new(600, 900);
        let main = vec![rectangle(150.0, 100.0, 200.0, 600.0)];
        let candidates = vec![
            circle(362.0, 300.0, 20.0),
            circle(138.0, 500.0, 20.0),
            circle(405.0, 500.0, 20.0),
            circle(390.0, 300.0, 20.0),
            rectangle(1.0, 1.0, 590.0, 890.0),
            rectangle(180.0, 180.0, 20.0, 20.0),
        ];
        let result =
            complete_attachments(main.clone(), &main, &candidates, &image, &[], None).unwrap();
        assert!(contains(&result, 378.0, 300.0));
        assert!(contains(&result, 120.0, 500.0));
        assert!(!contains(&result, 405.0, 500.0));
        assert!(!contains(&result, 405.0, 300.0));
        assert!(!contains(&result, 20.0, 20.0));
    }

    #[test]
    fn does_not_fill_tag_regions_or_bridge_a_gap_created_by_straightening() {
        let image = RgbImage::new(600, 900);
        let original = vec![rectangle(150.0, 100.0, 200.0, 600.0)];
        let parts = vec![circle(362.0, 300.0, 20.0)];
        let blocked = complete_attachments(
            original.clone(),
            &original,
            &parts,
            &image,
            &[(365, 290, 20, 20)],
            None,
        )
        .unwrap();
        assert!(!contains(&blocked, 378.0, 300.0));
        let moved = vec![rectangle(150.0, 100.0, 185.0, 600.0)];
        let disconnected =
            complete_attachments(moved, &original, &parts, &image, &[], None).unwrap();
        assert!(!contains(&disconnected, 378.0, 300.0));
    }
}
