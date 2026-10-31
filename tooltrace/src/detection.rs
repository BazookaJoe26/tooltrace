// AprilTag detection module
// Implements AprilTag detection using kornia-apriltag

use anyhow::{Context, Result};
use kornia_apriltag::{AprilTagDecoder, DecodeTagsConfig};
use kornia_apriltag::{decoder::{decode_tags, Detection, GrayModelPair}, quad::Quad, utils::Point2d};
use kornia_apriltag::family::TagFamilyKind;
use kornia_image::{Image, ImageSize};
use image::{GenericImageView, GrayImage, Rgb, RgbImage};
use imageproc::drawing::{draw_line_segment_mut, draw_filled_circle_mut, draw_text_mut};
use ab_glyph::{FontRef, PxScale};
use std::borrow::Cow;
use opencv::{core::{Mat, Point, Vector}, imgproc, prelude::*};

#[derive(Debug, Default, Clone, Copy)]
pub struct DetectionOptions {
    pub max_dimension: Option<u32>,
    pub smoothing: bool,
}

pub struct TagDetection {
    pub id: u32,
    pub center: (f64, f64),
    pub corners: [(f64, f64); 4],
    pub hamming_distance: u32,
    pub decision_margin: f64,
}

pub fn detect_apriltags(image_path: &str, options: DetectionOptions, debug_output: Option<&str>) -> Result<Vec<TagDetection>> {
    // Load the image using the image crate
    let img_rgb = image::open(image_path)
        .with_context(|| format!("Failed to read image: {}", image_path))?;

    let (orig_width, orig_height) = img_rgb.dimensions();
    println!("Loaded image: {}x{}", orig_width, orig_height);

    detect_apriltags_from_image(&img_rgb.to_rgb8(), options, debug_output)
}

/// Detect AprilTags from an in-memory image
pub fn detect_apriltags_from_image(img_rgb: &RgbImage, options: DetectionOptions, debug_output: Option<&str>) -> Result<Vec<TagDetection>> {
    let (orig_width, orig_height) = img_rgb.dimensions();
    println!("Processing image: {}x{}", orig_width, orig_height);

    let img_rgb_resized = prepare_detection_image(img_rgb, options.max_dimension)?;
    let (width, height) = img_rgb_resized.dimensions();
    if (width, height) == (orig_width, orig_height) {
        println!("Detecting at native resolution: {}x{}", width, height);
    } else {
        println!("Downsampling to {}x{} for detection", width, height);
    }

    // Use the actual rounded dimensions for each coordinate axis.
    let scale_x = orig_width as f64 / width as f64;
    let scale_y = orig_height as f64 / height as f64;

    // Do not silently halve the image again inside the decoder.
    let mut config = DecodeTagsConfig::new(vec![TagFamilyKind::Tag36H11]);
    config.downscale_factor = 1;
    config.min_tag_width = config.tag_families.iter()
        .map(|family| family.width_at_border).min().unwrap_or(3).max(3);

    let img_size = ImageSize {
        width: width as usize,
        height: height as usize,
    };

    let mut decoder = AprilTagDecoder::new(config, img_size)
        .context("Failed to create AprilTag decoder")?;

    let mut results = decode_pass(&mut decoder, &img_rgb_resized, scale_x, scale_y)?;
    println!("Found {} AprilTag(s) in first pass", results.len());

    if options.smoothing && !has_all_corner_tags(&results) {
        println!("Incomplete corner tag set; retrying with 1 px Gaussian smoothing");
        let smoothed = image::imageops::blur(img_rgb_resized.as_ref(), 1.0);
        decoder.clear();
        let smoothed_results = decode_pass(&mut decoder, &smoothed, scale_x, scale_y)?;
        println!("Found {} AprilTag(s) in smoothed pass", smoothed_results.len());
        merge_missing_tags(&mut results, smoothed_results);
    }

    println!("Found {} AprilTag(s) total", results.len());
    for det in &results {
        println!("  Tag ID {}: center=({:.1}, {:.1}), hamming={}, margin={:.2}",
            det.id, det.center.0, det.center.1, det.hamming_distance, det.decision_margin);
    }

    if let Some(debug_path) = debug_output {
        save_debug_image(img_rgb, &results, debug_path)?;
    }

    Ok(results)
}

fn prepare_detection_image(img: &RgbImage, max_dimension: Option<u32>) -> Result<Cow<'_, RgbImage>> {
    anyhow::ensure!(img.width() > 0 && img.height() > 0, "Detection image must not be empty");
    if let Some(max_dimension) = max_dimension {
        anyhow::ensure!(max_dimension > 0, "Detection maximum dimension must be positive");
        if img.width().max(img.height()) > max_dimension {
            let scale = max_dimension as f64 / img.width().max(img.height()) as f64;
            let width = (img.width() as f64 * scale).round().max(1.0) as u32;
            let height = (img.height() as f64 * scale).round().max(1.0) as u32;
            return Ok(Cow::Owned(image::imageops::resize(
                img, width, height, image::imageops::FilterType::Lanczos3,
            )));
        }
    }
    Ok(Cow::Borrowed(img))
}

fn has_all_corner_tags(detections: &[TagDetection]) -> bool {
    [0, 4, 8].into_iter().any(|base_id| {
        (base_id..base_id + 4).all(|id| detections.iter().any(|det| det.id == id))
    })
}

fn merge_missing_tags(results: &mut Vec<TagDetection>, smoothed: Vec<TagDetection>) {
    for tag in smoothed {
        if !results.iter().any(|existing| existing.id == tag.id) {
            results.push(tag);
        }
    }
}

fn decode_pass(decoder: &mut AprilTagDecoder, img_rgb: &RgbImage, scale_x: f64, scale_y: f64) -> Result<Vec<TagDetection>> {
    let img_gray = image::imageops::grayscale(img_rgb);
    let kornia_img = Image::new(
        ImageSize {
            width: img_gray.width() as usize,
            height: img_gray.height() as usize,
        },
        img_gray.as_raw().clone(),
        kornia_image::allocator::CpuAllocator,
    )?;
    let detections = decoder.decode(&kornia_img)
        .context("Failed to decode AprilTags")?;

    let mut results = map_detections(&detections, scale_x, scale_y);
    if !has_all_corner_tags(&results) {
        // Kornia's quad fitting can miss large photographed markers. Supplement
        // its candidates with native-resolution contours, then verify/decode
        // their bits and refine corners against the same grayscale pixels.
        println!("Incomplete corner tag set; checking native contour candidates (no resizing or smoothing)");
        let contour_tags = decode_contour_candidates(&img_gray, &kornia_img)?;
        merge_missing_tags(&mut results, map_detections(&contour_tags, scale_x, scale_y));
    }
    Ok(results)
}

fn decode_contour_candidates(gray: &GrayImage, source: &Image<u8, 1, kornia_image::allocator::CpuAllocator>) -> Result<Vec<Detection>> {
    let mat = Mat::from_slice(gray.as_raw())?;
    let mat = mat.reshape(1, gray.height() as i32)?;
    let mut binary = Mat::default();
    imgproc::threshold(&mat, &mut binary, 0.0, 255.0, imgproc::THRESH_BINARY_INV | imgproc::THRESH_OTSU)?;
    let mut contours: Vector<Vector<Point>> = Vector::new();
    imgproc::find_contours_def(&binary, &mut contours, imgproc::RETR_LIST, imgproc::CHAIN_APPROX_SIMPLE)?;

    let mut config = DecodeTagsConfig::new(vec![TagFamilyKind::Tag36H11]);
    config.downscale_factor = 1;
    let mut results: Vec<Detection> = Vec::new();
    for contour in contours {
        if imgproc::contour_area(&contour, false)? < 64.0 {
            continue;
        }
        let mut polygon: Vector<Point> = Vector::new();
        imgproc::approx_poly_dp(&contour, &mut polygon, imgproc::arc_length(&contour, true)? * 0.02, true)?;
        if polygon.len() != 4 || !imgproc::is_contour_convex(&polygon)? {
            continue;
        }
        // Kornia expects positive signed area in image coordinates. A mirrored
        // winding fails decoding even when the contour correctly outlines a tag.
        let mut points = polygon.to_vec();
        if imgproc::contour_area(&polygon, true)? < 0.0 {
            points.reverse();
        }
        let corners = std::array::from_fn(|i| Point2d {
            x: points[i].x as f32, y: points[i].y as f32,
        });
        let mut quad = Quad { corners, ..Default::default() };
        let mut gray_model = GrayModelPair::new();
        for tag in decode_tags(source, std::slice::from_mut(&mut quad), &mut config, &mut gray_model) {
            if !results.iter().any(|existing| existing.id == tag.id) {
                results.push(tag);
            }
        }
    }
    Ok(results)
}

fn map_detections(detections: &[Detection], scale_x: f64, scale_y: f64) -> Vec<TagDetection> {
    // Convert to our TagDetection format and scale back to original image coordinates
    detections.iter().map(|det| {
        // Extract corner positions from quad and scale back to original size
        let corners = [
            (det.quad.corners[0].x as f64 * scale_x, det.quad.corners[0].y as f64 * scale_y),
            (det.quad.corners[1].x as f64 * scale_x, det.quad.corners[1].y as f64 * scale_y),
            (det.quad.corners[2].x as f64 * scale_x, det.quad.corners[2].y as f64 * scale_y),
            (det.quad.corners[3].x as f64 * scale_x, det.quad.corners[3].y as f64 * scale_y),
        ];

        TagDetection {
            id: det.id as u32,
            center: projective_center(&corners),
            corners,
            hamming_distance: det.hamming as u32,
            decision_margin: det.decision_margin as f64,
        }
    }).collect()
}

/// A square's physical center projects to the intersection of its diagonals.
/// Averaging the four image corners introduces a bias under perspective.
fn projective_center(corners: &[(f64, f64); 4]) -> (f64, f64) {
    let (a, b) = (corners[0], corners[1]);
    let u = (corners[2].0 - a.0, corners[2].1 - a.1);
    let v = (corners[3].0 - b.0, corners[3].1 - b.1);
    let denominator = u.0 * v.1 - u.1 * v.0;
    if denominator.abs() <= f64::EPSILON {
        return (corners.iter().map(|p| p.0).sum::<f64>() / 4.0,
                corners.iter().map(|p| p.1).sum::<f64>() / 4.0);
    }
    let t = ((b.0 - a.0) * v.1 - (b.1 - a.1) * v.0) / denominator;
    (a.0 + t * u.0, a.1 + t * u.1)
}

fn save_debug_image(img: &RgbImage, detections: &[TagDetection], output_path: &str) -> Result<()> {
    let mut debug_img = img.clone();

    let rect_color = Rgb([0u8, 255u8, 0u8]); // Green rectangles
    let corner_color = Rgb([255u8, 0u8, 0u8]); // Red corners
    let center_color = Rgb([0u8, 0u8, 255u8]); // Blue center
    let text_color = Rgb([255u8, 255u8, 0u8]); // Yellow text

    // Use embedded Noto Sans font from ab_glyph
    let font = FontRef::try_from_slice(include_bytes!("../../fonts/NotoSans-Regular.ttf"))
        .context("Failed to load font")?;

    let scale = PxScale::from(40.0); // Font size

    for det in detections {
        // Draw lines connecting the corners to form the quad
        for i in 0..4 {
            let next = (i + 1) % 4;
            let p1 = (det.corners[i].0 as f32, det.corners[i].1 as f32);
            let p2 = (det.corners[next].0 as f32, det.corners[next].1 as f32);
            draw_line_segment_mut(&mut debug_img, p1, p2, rect_color);
        }

        // Draw corners as circles
        for corner in &det.corners {
            draw_filled_circle_mut(
                &mut debug_img,
                (corner.0 as i32, corner.1 as i32),
                8,
                corner_color
            );
        }

        // Draw center point
        draw_filled_circle_mut(
            &mut debug_img,
            (det.center.0 as i32, det.center.1 as i32),
            12,
            center_color
        );

        // Draw tag ID as text near the center
        let text = format!("ID {}", det.id);
        let text_x = (det.center.0 as i32) + 20; // Offset to the right
        let text_y = (det.center.1 as i32) - 20; // Offset up
        draw_text_mut(&mut debug_img, text_color, text_x, text_y, scale, &font, &text);

        println!("  Annotated Tag ID {} at ({:.1}, {:.1})", det.id, det.center.0, det.center.1);
    }

    // Save the annotated image
    debug_img.save(output_path)
        .with_context(|| format!("Failed to save debug image to {}", output_path))?;

    println!("Debug image saved to: {}", output_path);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contour_tags(img: &RgbImage) -> Vec<TagDetection> {
        let gray = image::imageops::grayscale(img);
        let size = ImageSize { width: gray.width() as usize, height: gray.height() as usize };
        let source = Image::new(size, gray.as_raw().clone(), kornia_image::allocator::CpuAllocator).unwrap();
        map_detections(&decode_contour_candidates(&gray, &source).unwrap(), 1.0, 1.0)
    }

    fn tag(id: u32, center: (f64, f64)) -> TagDetection {
        TagDetection {
            id, center, corners: [center; 4], hamming_distance: 0, decision_margin: 100.0,
        }
    }

    #[test]
    fn native_and_small_images_are_not_resampled() {
        let img = RgbImage::from_fn(123, 77, |x, y| Rgb([x as u8, y as u8, 127]));
        for limit in [None, Some(200)] {
            let prepared = prepare_detection_image(&img, limit).unwrap();
            assert!(matches!(prepared, Cow::Borrowed(_)));
            assert_eq!(prepared.as_ref(), &img);
        }
        assert!(prepare_detection_image(&img, Some(0)).is_err());
    }

    #[test]
    fn corner_set_requires_four_distinct_ids_from_one_paper() {
        for ids in [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]] {
            assert!(has_all_corner_tags(&ids.map(|id| tag(id, (0.0, 0.0)))));
        }
        for ids in [[0, 0, 0, 3], [0, 1, 2, 4]] {
            assert!(!has_all_corner_tags(&ids.map(|id| tag(id, (0.0, 0.0)))));
        }
    }

    #[test]
    fn smoothing_adds_missing_ids_without_replacing_original_coordinates() {
        let mut first = vec![tag(0, (10.0, 20.0)), tag(1, (30.0, 40.0))];
        merge_missing_tags(&mut first, vec![
            tag(0, (99.0, 99.0)), tag(2, (50.0, 60.0)), tag(3, (70.0, 80.0)),
        ]);
        assert_eq!(first.len(), 4);
        assert_eq!(first[0].center, (10.0, 20.0));
        assert!(has_all_corner_tags(&first));
    }

    #[test]
    fn native_contours_do_not_accept_unencoded_squares() {
        let mut img = RgbImage::from_pixel(200, 200, Rgb([255; 3]));
        for y in 50..150 {
            for x in 50..150 {
                img.put_pixel(x, y, Rgb([0; 3]));
            }
        }
        assert!(contour_tags(&img).is_empty());
    }

    #[test]
    fn center_is_perspective_projection_of_physical_square_center() {
        let project = |x: f64, y: f64| {
            let w = 1.0 + 0.004 * x + 0.002 * y;
            ((2.0 * x + 0.1 * y + 30.0) / w, (0.2 * x + 3.0 * y + 50.0) / w)
        };
        let corners = [project(0.0, 0.0), project(40.0, 0.0),
                       project(40.0, 40.0), project(0.0, 40.0)];
        let center = projective_center(&corners);
        let expected = project(20.0, 20.0);
        assert!((center.0 - expected.0).hypot(center.1 - expected.1) < 1e-10);
        let mean_x = corners.iter().map(|p| p.0).sum::<f64>() / 4.0;
        assert!((mean_x - expected.0).abs() > 1.0);
    }

    #[test]
    fn resized_tag_coordinates_map_back_to_original_image() {
        // An ideal four-tag sheet generated from the public 36h11 specification.
        let family: kornia_apriltag::family::TagFamily = TagFamilyKind::Tag36H11.into();
        let mut img = RgbImage::from_pixel(640, 900, Rgb([255; 3]));
        let origins = [(20, 20), (500, 20), (500, 760), (20, 760)];
        for (id, &(x, y)) in origins.iter().enumerate() {
            for py in 10..90 {
                for px in 10..90 {
                    img.put_pixel(x + px, y + py, Rgb([0; 3]));
                }
            }
            for bit in 0..family.nbits {
                if (family.code_data[id] >> (family.nbits - bit - 1)) & 1 != 0 {
                    let bx = (family.bit_x[bit] as u32 + 1) * 10;
                    let by = (family.bit_y[bit] as u32 + 1) * 10;
                    for py in 0..10 {
                        for px in 0..10 {
                            img.put_pixel(x + bx + px, y + by + py, Rgb([255; 3]));
                        }
                    }
                }
            }
        }
        // Exercise the alternative candidate finder at native multi-megapixel
        // resolution, including winding, bit validation and corner localization.
        let large = image::imageops::resize(&img, 2560, 3600, image::imageops::FilterType::Nearest);
        let contour_results = contour_tags(&large);
        assert_eq!(contour_results.len(), 4);
        for det in contour_results {
            let (x, y) = origins[det.id as usize];
            assert_eq!(det.hamming_distance, 0);
            assert!((det.center.0 - (x + 50) as f64 * 4.0).abs() < 2.0);
            assert!((det.center.1 - (y + 50) as f64 * 4.0).abs() < 2.0);
        }
        for limit in [None, Some(500)] {
            let results = detect_apriltags_from_image(&img, DetectionOptions {
                max_dimension: limit, smoothing: false,
            }, None).unwrap();
            assert_eq!(results.len(), 4);
            for det in results {
                let (x, y) = origins[det.id as usize];
                assert!((det.center.0 - (x + 50) as f64).abs() < 2.0);
                assert!((det.center.1 - (y + 50) as f64).abs() < 2.0);
            }
        }
    }
}
