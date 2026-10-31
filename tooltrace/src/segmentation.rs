// Object segmentation module
// Implements background subtraction and edge detection to find object contours

use anyhow::Result;
use image::RgbImage;
use opencv::{
    core::{Mat, MatTraitConst, Point, Scalar, Vector, BORDER_DEFAULT, AlgorithmHint},
    imgcodecs,
    imgproc::{CHAIN_APPROX_SIMPLE, RETR_EXTERNAL},
};
use tooltrace_common::{Contour, Point2DMm};

pub struct SegmentationResult {
    pub contours: Vec<Contour>,
    pub edge_based: bool,
}

/// Segment objects from the flattened paper image using FastSAM or edge detection fallback
/// Uses the installed model when available, otherwise an explicit edge fallback.
/// Excludes AprilTag regions and returns only outer boundary contours for tool storage plate generation
pub fn segment_object(
    flattened_image: &RgbImage,
    tag_regions: &[(i32, i32, i32, i32)], // (x, y, width, height) in pixels at 300 DPI
    model_override: Option<&std::path::Path>,
    debug_path: Option<&str>,
) -> Result<SegmentationResult> {
    // Try FastSAM first (best for segmenting all objects), fallback to edge detection
    if let Some(fastsam_path) = crate::fastsam::find_model(model_override)? {
        println!("Using FastSAM model for segmentation: {}", fastsam_path.display());
        return Ok(SegmentationResult {
            contours: segment_with_fastsam(flattened_image, &fastsam_path, tag_regions, debug_path)?,
            edge_based: false,
        });
    }

    println!("No ML models found, using edge detection fallback");
    Ok(SegmentationResult {
        contours: segment_with_edges(flattened_image, tag_regions, debug_path)?,
        edge_based: true,
    })
}

/// FastSAM based segmentation using ONNX Runtime
/// FastSAM (Fast Segment Anything Model) segments all objects in the image without class labels
fn segment_with_fastsam(
    flattened_image: &RgbImage,
    model_path: &std::path::Path,
    tag_regions: &[(i32, i32, i32, i32)],
    debug_path: Option<&str>,
) -> Result<Vec<Contour>> {
    use ort::session::Session;
    use ort::execution_providers::DirectMLExecutionProvider;
    use ort::value::TensorRef;
    use ndarray::Array;

    println!("Loading FastSAM model with ONNX Runtime from: {}", model_path.display());

    let img_height = flattened_image.height() as i32;
    let img_width = flattened_image.width() as i32;

    println!("Input image size: {}x{}", img_width, img_height);

    // Create ONNX Runtime session with GPU acceleration
    let build_session = |gpu: bool| -> ort::Result<Session> {
        let mut builder = Session::builder()?.with_parallel_execution(false)?.with_intra_threads(4)?;
        if gpu {
            builder = builder.with_memory_pattern(false)?.with_execution_providers([
                DirectMLExecutionProvider::default().build().error_on_failure(),
            ])?;
        }
        builder.commit_from_file(model_path)
    };
    let mut session = match build_session(true) {
        Ok(session) => { println!("FastSAM execution provider: DirectML"); session }
        Err(error) => { println!("DirectML unavailable ({error}); loading FastSAM on CPU"); build_session(false)? }
    };

    // Honor a static square input size; dynamic exports default to 1024.
    let input_shape = session.inputs.first().and_then(|i| i.input_type.tensor_shape())
        .ok_or_else(|| anyhow::anyhow!("FastSAM model needs an image tensor input"))?;
    if input_shape.len() != 4 || input_shape[1] != 3 || (input_shape[2] > 0 && input_shape[3] > 0 && input_shape[2] != input_shape[3]) {
        anyhow::bail!("Expected FastSAM input [1,3,size,size], got {input_shape:?}");
    }
    let input_size = if input_shape[2] > 0 { input_shape[2] as usize } else { 1024 };

    // Prepare input: resize and normalize
    let letterbox = crate::fastsam::Letterbox::new(flattened_image.width(), flattened_image.height(), input_size as u32);
    let resized = letterbox.prepare(flattened_image);
    println!("FastSAM input: {}x{}, aspect ratio preserved (padding {}, {})", input_size, input_size, letterbox.left, letterbox.top);

    // Convert to an RGB float32 BCHW tensor.
    let mut input_array = Array::zeros((1, 3, input_size, input_size));

    for y in 0..input_size {
        for x in 0..input_size {
            let pixel = resized.get_pixel(x as u32, y as u32);
            input_array[[0, 0, y, x]] = pixel[0] as f32 / 255.0; // R
            input_array[[0, 1, y, x]] = pixel[1] as f32 / 255.0; // G
            input_array[[0, 2, y, x]] = pixel[2] as f32 / 255.0; // B
        }
    }

    println!("Prepared input tensor: {:?}", input_array.shape());

    // Run inference
    println!("Running FastSAM inference with ONNX Runtime...");

    let input_tensor = TensorRef::from_array_view(&input_array)?;
    let input_name = session.inputs[0].name.clone();
    let outputs = session.run(ort::inputs![input_name.as_str() => input_tensor])?;

    println!("Got {} output tensors", outputs.len());

    // Debug: print all output shapes
    for (idx, (name, value)) in outputs.iter().enumerate() {
        if let Ok(tensor) = value.try_extract_tensor::<f32>() {
            println!("  Output {} '{}': shape {:?}", idx, name, tensor.0);
        }
    }

    // Process FastSAM outputs
    // Expected: output0 [1, 37, N] where N is number of anchors
    //           output1 [1, 32, H, W] mask prototypes

    // Create debug paths for segmentation overlays
    let contour_debug_path = debug_path.map(|p| {
        p.replace("_contours.jpg", "_fastsam_contours.jpg")
    });
    let masks_debug_path = debug_path.map(|p| {
        p.replace("_contours.jpg", "_fastsam_masks.jpg")
    });

    // Convert source image to OpenCV Mat for debug overlay
    let source_mat = if contour_debug_path.is_some() || masks_debug_path.is_some() {
        let img_data: Vec<u8> = flattened_image.as_raw().clone();
        let mat = Mat::from_slice(&img_data)?;
        let rgb = mat.reshape(3, img_height)?;
        let mut bgr = Mat::default();
        opencv::imgproc::cvt_color_def(&rgb, &mut bgr, opencv::imgproc::COLOR_RGB2BGR)?;
        Some(bgr)
    } else {
        None
    };

    let contours = process_fastsam_outputs(
        &outputs,
        img_width,
        img_height,
        letterbox,
        tag_regions,
        contour_debug_path.as_deref(),
        masks_debug_path.as_deref(),
        source_mat.as_ref(),
    )?;

    Ok(contours)
}

/// Calculate IoU (Intersection over Union) for two boxes
fn calculate_iou(box_a: (f32, f32, f32, f32), box_b: (f32, f32, f32, f32)) -> f32 {
    let (ax1, ay1, aw, ah) = box_a;
    let (bx1, by1, bw, bh) = box_b;

    let ax2 = ax1 + aw;
    let ay2 = ay1 + ah;
    let bx2 = bx1 + bw;
    let by2 = by1 + bh;

    let intersect_x1 = ax1.max(bx1);
    let intersect_y1 = ay1.max(by1);
    let intersect_x2 = ax2.min(bx2);
    let intersect_y2 = ay2.min(by2);

    if intersect_x1 >= intersect_x2 || intersect_y1 >= intersect_y2 {
        return 0.0;
    }

    let intersection = (intersect_x2 - intersect_x1) * (intersect_y2 - intersect_y1);
    let area_a = aw * ah;
    let area_b = bw * bh;
    let union = area_a + area_b - intersection;

    intersection / union
}

/// Edge-based segmentation (fallback)
fn segment_with_edges(
    flattened_image: &RgbImage,
    tag_regions: &[(i32, i32, i32, i32)],
    debug_path: Option<&str>,
) -> Result<Vec<Contour>> {
    println!("Segmenting objects with {} AprilTag exclusion region(s)...", tag_regions.len());

    // Convert image to OpenCV Mat
    let img_data: Vec<u8> = flattened_image.as_raw().clone();
    let mat = Mat::from_slice(&img_data)?;
    let mat = mat.reshape(3, flattened_image.height() as i32)?;

    // Convert to grayscale
    let mut gray = Mat::default();
    opencv::imgproc::cvt_color(&mat, &mut gray, opencv::imgproc::COLOR_RGB2GRAY, 0, AlgorithmHint::ALGO_HINT_DEFAULT)?;

    // Mask out AprilTag regions by drawing filled white rectangles (paper background color)
    for (x, y, w, h) in tag_regions {
        let rect = opencv::core::Rect::new(*x, *y, *w, *h);
        opencv::imgproc::rectangle(
            &mut gray,
            rect,
            Scalar::all(255.0), // White
            -1, // Filled
            opencv::imgproc::LINE_8,
            0,
        )?;

        println!("  Masking AprilTag region at ({}, {}) size {}x{}", x, y, w, h);
    }

    // Apply Gaussian blur to reduce noise
    let mut blurred = Mat::default();
    opencv::imgproc::gaussian_blur(
        &gray,
        &mut blurred,
        opencv::core::Size::new(5, 5),
        1.5,
        1.5,
        BORDER_DEFAULT,
        AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    // Apply Canny edge detection with MORE SENSITIVE thresholds for complete tool outlines
    // Lower thresholds (20, 60) instead of (50, 150) to capture subtle edges
    let mut edges = Mat::default();
    opencv::imgproc::canny(&blurred, &mut edges, 20.0, 60.0, 3, false)?;

    println!("Applied sensitive Canny edge detection (20, 60 thresholds)");

    // Dilate edges to close small gaps
    let kernel = opencv::imgproc::get_structuring_element(
        opencv::imgproc::MORPH_RECT,
        opencv::core::Size::new(3, 3),
        opencv::core::Point::new(-1, -1),
    )?;
    let mut dilated = Mat::default();
    opencv::imgproc::dilate(
        &edges,
        &mut dilated,
        &kernel,
        opencv::core::Point::new(-1, -1),
        2,
        BORDER_DEFAULT,
        Scalar::default(),
    )?;

    // Find contours
    let mut contours = Vector::<Vector<Point>>::new();
    opencv::imgproc::find_contours(
        &dilated,
        &mut contours,
        RETR_EXTERNAL,
        CHAIN_APPROX_SIMPLE,
        Point::new(0, 0),
    )?;

    println!("Found {} contours", contours.len());

    // Filter contours by area (remove very small noise)
    // Lower threshold to capture smaller tools
    let min_area = 500.0; // Minimum area in pixels (~3mm² at 300 DPI)
    let mut filtered_contours = Vec::new();

    for i in 0..contours.len() {
        let contour = contours.get(i)?;
        let area = opencv::imgproc::contour_area(&contour, false)?;

        if area > min_area {
            // Calculate bounding box to check if it overlaps with AprilTag regions
            let bounding_rect = opencv::imgproc::bounding_rect(&contour)?;

            // Check if contour overlaps significantly with any tag region
            let mut overlaps_tag = false;
            for (tx, ty, tw, th) in tag_regions {
                let tag_rect = opencv::core::Rect::new(*tx, *ty, *tw, *th);
                let intersection = bounding_rect & tag_rect;

                // If intersection area is > 10% of contour bounding box, it's likely a tag edge
                let intersection_area = (intersection.width * intersection.height) as f64;
                let contour_bbox_area = (bounding_rect.width * bounding_rect.height) as f64;

                if intersection_area > contour_bbox_area * 0.1 {
                    overlaps_tag = true;
                    break;
                }
            }

            if !overlaps_tag {
                println!("  Contour {}: area = {:.1} pixels ({:.1}mm²)", i, area, area / (11.811 * 11.811));

                // Convert OpenCV contour to our Contour type (in pixels for now)
                let mut points = Vec::new();
                for j in 0..contour.len() {
                    let pt = contour.get(j)?;
                    points.push(Point2DMm {
                        x: pt.x as f64,
                        y: pt.y as f64,
                    });
                }

                filtered_contours.push(Contour {
                    points,
                    closed: true, // RETR_EXTERNAL always gives closed outer boundaries
                });
            } else {
                println!("  Contour {}: EXCLUDED (overlaps with AprilTag region)", i);
            }
        }
    }

    println!("Filtered to {} tool contours (area > {} pixels, excluding tags)", filtered_contours.len(), min_area);

    // Save debug visualization if requested
    if let Some(path) = debug_path {
        let mut debug_img = mat.try_clone()?;
        let color = Scalar::new(0.0, 255.0, 0.0, 0.0); // Green

        for i in 0..contours.len() {
            let contour = contours.get(i)?;
            let area = opencv::imgproc::contour_area(&contour, false)?;
            if area > min_area {
                let mut contour_vec: Vector<Vector<Point>> = Vector::new();
                contour_vec.push(contour);
                opencv::imgproc::draw_contours(
                    &mut debug_img,
                    &contour_vec,
                    0,
                    color,
                    2,
                    opencv::imgproc::LINE_8,
                    &Mat::default(),
                    i32::MAX,
                    Point::new(0, 0),
                )?;
            }
        }

        imgcodecs::imwrite(path, &debug_img, &Vector::new())?;
        println!("Saved contour debug image to: {}", path);
    }

    Ok(filtered_contours)
}

/// Process FastSAM ONNX Runtime outputs to extract contours from segmentation masks
fn process_fastsam_outputs(
    outputs: &ort::session::SessionOutputs,
    img_width: i32,
    img_height: i32,
    letterbox: crate::fastsam::Letterbox,
    tag_regions: &[(i32, i32, i32, i32)],
    contour_debug_path: Option<&str>,
    masks_debug_path: Option<&str>,
    source_image: Option<&Mat>,
) -> Result<Vec<Contour>> {
    use ndarray::{ArrayView3, ArrayView4};

    println!("Processing FastSAM ONNX Runtime outputs with mask extraction...");

    // FastSAM outputs:
    // output0: shape [1, 37, N] - detections (4 bbox + 1 obj + 32 mask coeffs)
    // output1: shape [1, 32, H, W] - mask prototypes

    // Tunable parameters for FastSAM segmentation
    let conf_threshold = 0.40f32;      // FastSAM README defaults
    let nms_threshold = 0.90f32;
    let mask_threshold = 0.0f32;       // Native mask logits > 0 (sigmoid > 0.5)

    // Area-based filtering to remove background and noise
    let total_pixels = (img_width * img_height) as f32;
    let min_area_pixels = total_pixels * 0.0005;  // Min 0.05% of image (filter noise)
    let max_area_pixels = total_pixels * 0.70;    // Max 70% of image (filter paper background)

    println!("Area filtering: min={:.0} pixels, max={:.0} pixels", min_area_pixels, max_area_pixels);

    // Extract detection tensor (output0)
    if outputs.len() != 2 { anyhow::bail!("FastSAM requires detection and mask-prototype outputs"); }
    let det_value = &outputs[0];
    let det_tensor = det_value.try_extract_tensor::<f32>()?;
    let det_shape = det_tensor.0;
    let det_data = det_tensor.1;

    println!("Detections shape: {:?}", det_shape);

    // Extract mask prototypes (output1)
    let proto_value = &outputs[1];
    let proto_tensor = proto_value.try_extract_tensor::<f32>()?;
    let proto_shape = proto_tensor.0;
    let proto_data = proto_tensor.1;

    println!("Mask prototypes shape: {:?}", proto_shape);

    if det_shape.len() != 3 || det_shape[0] != 1 || det_shape[1] != 37 || proto_shape.len() != 4 || proto_shape[0] != 1 || proto_shape[1] != 32 {
        anyhow::bail!("Expected FastSAM outputs [1,37,N] and [1,32,H,W], got {det_shape:?} and {proto_shape:?}");
    }

    // Create ArrayViews
    let det_view = ArrayView3::from_shape(
        (det_shape[0] as usize, det_shape[1] as usize, det_shape[2] as usize),
        det_data,
    )?;

    let proto_view = ArrayView4::from_shape(
        (proto_shape[0] as usize, proto_shape[1] as usize, proto_shape[2] as usize, proto_shape[3] as usize),
        proto_data,
    )?;

    let num_detections = det_shape[2] as usize;
    let proto_h = proto_shape[2] as usize;
    let proto_w = proto_shape[3] as usize;

    println!("Processing {} potential detections", num_detections);
    println!("Mask prototype size: {}x{}", proto_h, proto_w);

    // Collect detections above threshold
    struct Detection {
        bbox: (f32, f32, f32, f32), // x, y, w, h in image coordinates
        confidence: f32,
        mask_coeffs: Vec<f32>, // 32 mask coefficients
        det_idx: usize, // Original detection index
    }

    let mut detections_list = Vec::new();

    // Process each detection
    for i in 0..num_detections {
        // Extract box coordinates [cx, cy, w, h] (indices 0-3)
        let (cx, cy) = letterbox.point_to_original(det_view[[0, 0, i]], det_view[[0, 1, i]]);
        let (w, h) = letterbox.size_to_original(det_view[[0, 2, i]], det_view[[0, 3, i]]);

        // Extract objectness score (index 4)
        let objectness = det_view[[0, 4, i]];

        if objectness.is_finite() && objectness > conf_threshold && [cx, cy, w, h].iter().all(|v| v.is_finite()) && w > 0.0 && h > 0.0 {
            // Extract 32 mask coefficients (indices 5-36)
            let mut mask_coeffs = Vec::with_capacity(32);
            for j in 0..32 {
                mask_coeffs.push(det_view[[0, 5 + j, i]]);
            }

            detections_list.push(Detection {
                bbox: (cx, cy, w, h),
                confidence: objectness,
                mask_coeffs,
                det_idx: i,
            });
        }
    }

    println!("Found {} detections above confidence threshold", detections_list.len());

    // Simple NMS implementation
    let mut keep_indices = Vec::new();
    let mut sorted_indices: Vec<usize> = (0..detections_list.len()).collect();
    sorted_indices.sort_by(|&a, &b| {
        detections_list[b].confidence.partial_cmp(&detections_list[a].confidence).unwrap()
    });

    let mut suppressed = vec![false; detections_list.len()];

    for &idx in &sorted_indices {
        if suppressed[idx] {
            continue;
        }

        keep_indices.push(idx);

        let det_a = &detections_list[idx];
        let box_a = (
            det_a.bbox.0 - det_a.bbox.2 / 2.0,
            det_a.bbox.1 - det_a.bbox.3 / 2.0,
            det_a.bbox.2,
            det_a.bbox.3,
        );

        for &other_idx in &sorted_indices {
            if other_idx == idx || suppressed[other_idx] {
                continue;
            }

            let det_b = &detections_list[other_idx];
            let box_b = (
                det_b.bbox.0 - det_b.bbox.2 / 2.0,
                det_b.bbox.1 - det_b.bbox.3 / 2.0,
                det_b.bbox.2,
                det_b.bbox.3,
            );

            let iou = calculate_iou(box_a, box_b);
            if iou > nms_threshold {
                suppressed[other_idx] = true;
            }
        }
    }

    println!("After NMS: {} final detections", keep_indices.len());

    // Create debug images if requested
    // 1. Contour overlay on source image
    let mut contour_debug_img = if contour_debug_path.is_some() {
        if let Some(src) = source_image {
            src.try_clone()?
        } else {
            Mat::default()
        }
    } else {
        Mat::default()
    };

    // 2. Mask overlay showing segmentation regions
    let mut masks_debug_img = if masks_debug_path.is_some() {
        if let Some(src) = source_image {
            src.try_clone()?
        } else {
            Mat::default()
        }
    } else {
        Mat::default()
    };

    // Process each kept detection to generate masks and extract contours
    let mut contours_result = Vec::new();
    let mut detection_count = 0;

    for &idx in &keep_indices {
        let det = &detections_list[idx];

        println!("Processing detection {}: conf={:.2}", detection_count, det.confidence);

        // Convert to x1, y1, x2, y2
        let x1 = (det.bbox.0 - det.bbox.2 / 2.0).max(0.0) as i32;
        let y1 = (det.bbox.1 - det.bbox.3 / 2.0).max(0.0) as i32;
        let x2 = (det.bbox.0 + det.bbox.2 / 2.0).min(img_width as f32) as i32;
        let y2 = (det.bbox.1 + det.bbox.3 / 2.0).min(img_height as f32) as i32;
        if x2 <= x1 || y2 <= y1 { continue; }

        // Check if overlaps with AprilTag regions
        let mut overlaps_tag = false;
        for (tx, ty, tw, th) in tag_regions {
            let tag_x2 = tx + tw;
            let tag_y2 = ty + th;

            let intersect_x1 = x1.max(*tx);
            let intersect_y1 = y1.max(*ty);
            let intersect_x2 = x2.min(tag_x2);
            let intersect_y2 = y2.min(tag_y2);

            if intersect_x1 < intersect_x2 && intersect_y1 < intersect_y2 {
                let intersection_area = ((intersect_x2 - intersect_x1) * (intersect_y2 - intersect_y1)) as f64;
                let bbox_area = ((x2 - x1) * (y2 - y1)) as f64;

                if intersection_area > bbox_area * 0.1 {
                    overlaps_tag = true;
                    break;
                }
            }
        }

        if overlaps_tag {
            println!("  Excluded: overlaps with AprilTag region");
            continue;
        }

        // Generate segmentation mask from coefficients and prototypes
        // Mask logits = sum(coeffs[i] * prototypes[i]).
        let mut mask = vec![0.0f32; proto_h * proto_w];

        for proto_idx in 0..32 {
            let coeff = det.mask_coeffs[proto_idx];
            for y in 0..proto_h {
                for x in 0..proto_w {
                    let proto_val = proto_view[[0, proto_idx, y, x]];
                    mask[y * proto_w + x] += coeff * proto_val;
                }
            }
        }

        // Debug: Check mask values before sigmoid
        let mask_min = mask.iter().copied().fold(f32::INFINITY, f32::min);
        let mask_max = mask.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        println!("  Mask before sigmoid: min={:.3}, max={:.3}", mask_min, mask_max);

        // Restore logits at original resolution before thresholding, matching
        // retina-mask decoding. Remove the input padding before resizing.
        let mask_mat_temp = Mat::from_slice(&mask)?;
        let mask_mat = mask_mat_temp.reshape(1, proto_h as i32)?;
        let mut resized_mask = letterbox.restore_mask(&mask_mat)?;
        // A prototype describes the whole scene. Each instance is restricted
        // to its own detected box, so unrelated positive regions cannot leak in.
        for (index, value) in opencv::prelude::MatTraitManual::data_typed_mut::<f32>(&mut resized_mask)?.iter_mut().enumerate() {
            let x = index as i32 % img_width;
            let y = index as i32 / img_width;
            if x < x1 || x >= x2 || y < y1 || y >= y2 { *value = -100.0; }
        }

        // Debug: Check resized mask values
        let mut min_val = 0.0;
        let mut max_val = 0.0;
        opencv::core::min_max_loc(
            &resized_mask,
            Some(&mut min_val),
            Some(&mut max_val),
            None,
            None,
            &Mat::default(),
        )?;
        println!("  Resized mask: min={:.3}, max={:.3}", min_val, max_val);

        // Threshold native-resolution logits at zero (probability 0.5).
        let mut binary_mask = Mat::default();
        opencv::imgproc::threshold(
            &resized_mask,
            &mut binary_mask,
            mask_threshold as f64,
            1.0,                     // Max value is 1.0
            opencv::imgproc::THRESH_BINARY,
        )?;

        // Count pixels above threshold
        let nonzero = opencv::core::count_non_zero(&binary_mask)?;
        println!("  Binary mask: {} pixels above threshold ({}%)",
                 nonzero, (nonzero as f32 * 100.0) / (img_width * img_height) as f32);

        // Convert to 8-bit for contour detection
        let mut mask_8u = Mat::default();
        binary_mask.convert_to(&mut mask_8u, opencv::core::CV_8U, 1.0, 0.0)?;

        // Draw segmentation mask on mask debug image if requested
        if masks_debug_path.is_some() {
            // Generate a unique color for this detection (BGR format)
            let color = Scalar::new(
                ((detection_count * 70) % 256) as f64,
                ((detection_count * 150) % 256) as f64,
                ((detection_count * 230) % 256) as f64,
                0.0,
            );

            // Create a colored overlay where mask is active
            let mut colored_overlay = Mat::default();
            masks_debug_img.try_clone()?.copy_to(&mut colored_overlay)?;

            // Fill mask region with color using copyTo with mask
            let mut color_mat = Mat::new_rows_cols_with_default(
                img_height,
                img_width,
                opencv::core::CV_8UC3,
                color,
            )?;

            // Blend colored region with original image (60% color, 40% original)
            let mut temp = Mat::default();
            opencv::core::add_weighted(
                &color_mat,
                0.6,
                &colored_overlay,
                0.4,
                0.0,
                &mut temp,
                -1,
            )?;

            // Copy only the masked region to the debug image
            temp.copy_to_masked(&mut masks_debug_img, &mask_8u)?;
        }

        // Find contours from binary mask
        let mut mask_contours = Vector::<Vector<Point>>::new();
        opencv::imgproc::find_contours(
            &mask_8u,
            &mut mask_contours,
            RETR_EXTERNAL,
            CHAIN_APPROX_SIMPLE,
            Point::new(0, 0),
        )?;

        // Find the largest contour (main object)
        let mut largest_contour_idx = None;
        let mut largest_area = 0.0;

        for i in 0..mask_contours.len() {
            let contour = mask_contours.get(i)?;
            let area = opencv::imgproc::contour_area(&contour, false)?;
            if area > largest_area {
                largest_area = area;
                largest_contour_idx = Some(i);
            }
        }

        if let Some(contour_idx) = largest_contour_idx {
            let contour = mask_contours.get(contour_idx)?;

            // Filter by area to remove background (too large) and noise (too small)
            if largest_area < min_area_pixels as f64 {
                println!("  Skipped: area {:.1} pixels too small (min {:.0})", largest_area, min_area_pixels);
                continue;
            }
            if largest_area > max_area_pixels as f64 {
                println!("  Skipped: area {:.1} pixels too large (max {:.0}), likely background", largest_area, max_area_pixels);
                continue;
            }

            // Convert OpenCV contour to our Contour type
            let mut points = Vec::new();
            for j in 0..contour.len() {
                let pt = contour.get(j)?;
                points.push(Point2DMm {
                    x: pt.x as f64,
                    y: pt.y as f64,
                });
            }

            println!("  ✓ Extracted contour with {} points, area={:.1} pixels", points.len(), largest_area);

            contours_result.push(Contour {
                points,
                closed: true,
            });

            // Draw contour on contour debug image
            if contour_debug_path.is_some() {
                // Generate color for this detection
                let color = Scalar::new(
                    ((detection_count * 50) % 255) as f64,
                    ((detection_count * 100) % 255) as f64,
                    ((detection_count * 150) % 255) as f64,
                    0.0,
                );

                let mut contour_vec: Vector<Vector<Point>> = Vector::new();
                contour_vec.push(contour);
                opencv::imgproc::draw_contours(
                    &mut contour_debug_img,
                    &contour_vec,
                    0,
                    color,
                    2,
                    opencv::imgproc::LINE_8,
                    &Mat::default(),
                    i32::MAX,
                    Point::new(0, 0),
                )?;
            }

            detection_count += 1;
        } else {
            println!("  No valid contour found in mask");
        }
    }

    // Save debug images if requested
    if let Some(path) = contour_debug_path {
        imgcodecs::imwrite(path, &contour_debug_img, &Vector::new())?;
        println!("Saved contour overlay to: {}", path);
    }

    if let Some(path) = masks_debug_path {
        imgcodecs::imwrite(path, &masks_debug_img, &Vector::new())?;
        println!("Saved segmentation masks overlay to: {}", path);
    }

    // Remove contours that are fully contained within other contours
    let filtered_contours = remove_nested_contours(contours_result, img_width, img_height)?;

    println!("Returning {} contours after removing nested contours", filtered_contours.len());

    Ok(filtered_contours)
}

/// Remove contours that are fully contained within other contours
/// Uses OpenCV's pointPolygonTest to check containment
fn remove_nested_contours(
    contours: Vec<Contour>,
    img_width: i32,
    img_height: i32,
) -> Result<Vec<Contour>> {
    use opencv::core::Point;
    use opencv::core::Vector;
    use opencv::imgproc;

    if contours.len() <= 1 {
        return Ok(contours);
    }

    println!("Checking {} contours for containment...", contours.len());

    // Convert our contours to OpenCV format for polygon testing
    let mut opencv_contours: Vec<Vector<Point>> = Vec::new();
    for contour in &contours {
        let mut pts: Vector<Point> = Vector::new();
        for pt in &contour.points {
            pts.push(Point::new(pt.x as i32, pt.y as i32));
        }
        opencv_contours.push(pts);
    }

    // Track which contours are contained within others
    let mut is_contained = vec![false; contours.len()];

    // Check each pair of contours
    for i in 0..contours.len() {
        if is_contained[i] {
            continue; // Skip if already marked as contained
        }

        for j in 0..contours.len() {
            if i == j || is_contained[j] {
                continue;
            }

            // Check if contour i is fully inside contour j
            if is_contour_inside(&opencv_contours[i], &opencv_contours[j])? {
                is_contained[i] = true;
                println!("  Contour {} is contained within contour {}", i, j);
                break;
            }
        }
    }

    // Collect only non-contained contours
    let result: Vec<Contour> = contours
        .into_iter()
        .enumerate()
        .filter_map(|(idx, contour)| {
            if is_contained[idx] {
                None
            } else {
                Some(contour)
            }
        })
        .collect();

    let removed = opencv_contours.len() - result.len();
    if removed > 0 {
        println!("Removed {} nested contour(s)", removed);
    }

    Ok(result)
}

/// Check if contour A is fully contained inside contour B
/// Samples points from contour A and tests if they're all inside B
fn is_contour_inside(contour_a: &Vector<Point>, contour_b: &Vector<Point>) -> Result<bool> {
    use opencv::imgproc;

    if contour_a.is_empty() || contour_b.is_empty() {
        return Ok(false);
    }

    // Sample every Nth point from contour A to test (for efficiency)
    // For small contours, test all points; for large ones, sample
    let sample_step = if contour_a.len() < 50 {
        1
    } else {
        contour_a.len() / 50
    };

    let mut points_tested = 0;
    let mut points_inside = 0;

    for i in (0..contour_a.len()).step_by(sample_step) {
        let pt = contour_a.get(i)?;
        let point_f64 = opencv::core::Point2f::new(pt.x as f32, pt.y as f32);

        // pointPolygonTest returns:
        // > 0 if point is inside
        // < 0 if point is outside
        // = 0 if point is on edge
        let dist = imgproc::point_polygon_test(contour_b, point_f64, false)?;

        points_tested += 1;
        if dist >= 0.0 {
            points_inside += 1;
        }
    }

    // Contour A is inside B if all (or nearly all) sampled points are inside
    // Use 95% threshold to handle edge cases and floating point precision
    let inside_ratio = points_inside as f64 / points_tested as f64;
    Ok(inside_ratio >= 0.95)
}

/// Convert contours from pixels to millimeters
/// The flattened image is at 300 DPI
pub fn pixels_to_mm(contours: Vec<Contour>, dpi: f64) -> Vec<Contour> {
    let pixels_per_mm = dpi / 25.4; // 300 DPI = 11.811 pixels/mm

    contours.into_iter().map(|contour| {
        let points = contour.points.into_iter().map(|pt| {
            Point2DMm {
                x: pt.x / pixels_per_mm,
                y: pt.y / pixels_per_mm,
            }
        }).collect();

        Contour {
            points,
            closed: contour.closed,
        }
    }).collect()
}

/// Filter contours to keep only the best object contour using smart scoring
/// Considers area, centrality and edge proximity without rejecting elongated tools.
/// Returns a vector with at most one contour (the best object)
pub fn filter_largest_contour(
    contours: Vec<Contour>,
    debug_path: Option<&str>,
    source_image: &RgbImage,
) -> Result<Vec<Contour>> {
    if contours.is_empty() {
        println!("No contours to filter");
        return Ok(vec![]);
    }

    println!("Filtering {} contours using smart object scoring...", contours.len());

    let img_width = source_image.width() as f64;
    let img_height = source_image.height() as f64;
    let img_center_x = img_width / 2.0;
    let img_center_y = img_height / 2.0;

    // Calculate scores for each contour
    let mut contour_scores: Vec<(usize, f64, ContourMetrics)> = Vec::new();

    for (idx, contour) in contours.iter().enumerate() {
        if contour.points.len() < 3 || contour.points.iter().any(|p| !p.x.is_finite() || !p.y.is_finite()) {
            continue;
        }
        let metrics = calculate_contour_metrics(&contour.points, img_width, img_height);
        if metrics.area <= f64::EPSILON {
            continue;
        }
        let bbox_width = metrics.bbox.2 - metrics.bbox.0;
        let bbox_height = metrics.bbox.3 - metrics.bbox.1;
        if bbox_width >= img_width * 0.95 && bbox_height >= img_height * 0.95 {
            println!("  Contour {}: REJECTED (spans the paper boundary)", idx);
            continue;
        }

        // Calculate composite score
        let area_score = calculate_area_score(metrics.area, img_width * img_height);
        let centrality_score = calculate_centrality_score(
            metrics.centroid_x,
            metrics.centroid_y,
            img_center_x,
            img_center_y,
            img_width,
            img_height
        );
        let edge_proximity_score = calculate_edge_proximity_score(
            &metrics.bbox,
            img_width,
            img_height
        );

        // Circularity is not an object-quality measure: calipers, rulers and
        // other long/concave tools naturally have low circularity. Prioritize
        // substantial foreground area over small round printed scale glyphs.
        let total_score =
            area_score * 0.60 +
            centrality_score * 0.25 +
            edge_proximity_score * 0.15;

        contour_scores.push((idx, total_score, metrics));

        println!("  Contour {}: area={:.1}px², center=({:.0},{:.0}), compact={:.3}, edge_dist={:.1}px, SCORE={:.3}",
            idx, metrics.area, metrics.centroid_x, metrics.centroid_y,
            metrics.compactness, edge_proximity_score, total_score);
    }

    // Find the best scoring contour
    let Some((best_idx, best_score, best_metrics)) = contour_scores
        .iter()
        .max_by(|a, b| a.1.total_cmp(&b.1)) else {
            println!("No valid object contours after filtering");
            return Ok(vec![]);
        };

    println!("Best contour: index {} with score {:.3} (area={:.1}px², center=({:.0},{:.0}))",
        best_idx, best_score, best_metrics.area, best_metrics.centroid_x, best_metrics.centroid_y);

    // Save debug visualization if requested
    if let Some(path) = debug_path {
        save_best_contour_debug(&contours, *best_idx, source_image, path)?;
    }

    // Return only the best contour
    Ok(vec![contours[*best_idx].clone()])
}

/// Refine an edge-derived outline using foreground/background colors. The edge
/// contour seeds the mask; ambiguous boundary pixels can move in either direction
/// while printed interior details no longer become deep exterior indentations.
pub fn refine_edge_contours(
    contours: Vec<Contour>,
    source_image: &RgbImage,
    tag_regions: &[(i32, i32, i32, i32)],
    debug_path: Option<&str>,
) -> Result<Vec<Contour>> {
    let mut refined = Vec::new();
    for contour in contours {
        refined.push(refine_edge_contour(&contour, source_image, tag_regions)?);
    }
    if let Some(path) = debug_path {
        if !refined.is_empty() {
            save_best_contour_debug(&refined, 0, source_image, path)?;
        }
    }
    Ok(refined)
}

fn refine_edge_contour(
    contour: &Contour,
    source_image: &RgbImage,
    tag_regions: &[(i32, i32, i32, i32)],
) -> Result<Contour> {
    use opencv::{core::{Rect, Size, CV_8UC1}, prelude::*};
    let polygon: Vector<Point> = contour.points.iter()
        .map(|p| Point::new(p.x.round() as i32, p.y.round() as i32)).collect();
    let bounds = opencv::imgproc::bounding_rect(&polygon)?;
    // Work on an 8 mm neighborhood at the calibrated 300 DPI, keeping native pixels.
    let padding = (8.0_f64 * 300.0 / 25.4).round() as i32;
    let x = (bounds.x - padding).max(0);
    let y = (bounds.y - padding).max(0);
    let right = (bounds.x + bounds.width + padding).min(source_image.width() as i32);
    let bottom = (bounds.y + bounds.height + padding).min(source_image.height() as i32);
    let roi = Rect::new(x, y, right - x, bottom - y);
    if roi.width < 3 || roi.height < 3 {
        return Ok(contour.clone());
    }
    let mat = Mat::from_slice(source_image.as_raw())?;
    let mat = mat.reshape(3, source_image.height() as i32)?;
    let crop = Mat::roi(&mat, roi)?.try_clone()?;

    let local_polygon: Vector<Point> = polygon.iter().map(|p| Point::new(p.x - x, p.y - y)).collect();
    let local_contours = Vector::<Vector<Point>>::from_iter([local_polygon]);
    let mut filled = Mat::new_rows_cols_with_default(roi.height, roi.width, CV_8UC1, Scalar::all(0.0))?;
    opencv::imgproc::draw_contours(&mut filled, &local_contours, 0, Scalar::all(255.0), -1,
        opencv::imgproc::LINE_8, &Mat::default(), i32::MAX, Point::new(0, 0))?;
    let mut labels = Mat::new_rows_cols_with_default(roi.height, roi.width, CV_8UC1,
        Scalar::all(opencv::imgproc::GC_PR_BGD as f64))?;
    labels.set_to(&Scalar::all(opencv::imgproc::GC_PR_FGD as f64), &filled)?;
    let erosion_kernel = opencv::imgproc::get_structuring_element(opencv::imgproc::MORPH_ELLIPSE,
        Size::new(9, 9), Point::new(-1, -1))?;
    let mut interior = Mat::default();
    opencv::imgproc::erode(&filled, &mut interior, &erosion_kernel, Point::new(-1, -1), 1,
        BORDER_DEFAULT, Scalar::default())?;
    labels.set_to(&Scalar::all(opencv::imgproc::GC_FGD as f64), &interior)?;
    opencv::imgproc::rectangle(&mut labels, Rect::new(0, 0, roi.width, roi.height),
        Scalar::all(opencv::imgproc::GC_BGD as f64), 2, opencv::imgproc::LINE_8, 0)?;
    for &(tx, ty, tw, th) in tag_regions {
        let overlap = Rect::new(tx, ty, tw, th) & roi;
        if overlap.area() > 0 {
            opencv::imgproc::rectangle(&mut labels,
                Rect::new(overlap.x - x, overlap.y - y, overlap.width, overlap.height),
                Scalar::all(opencv::imgproc::GC_BGD as f64), -1, opencv::imgproc::LINE_8, 0)?;
        }
    }
    opencv::imgproc::grab_cut(&crop, &mut labels, Rect::default(), &mut Mat::default(),
        &mut Mat::default(), 5, opencv::imgproc::GC_INIT_WITH_MASK)?;
    let foreground: Vec<u8> = labels.data_typed::<u8>()?.iter().map(|&v| {
        if v == opencv::imgproc::GC_FGD as u8 || v == opencv::imgproc::GC_PR_FGD as u8 { 255 } else { 0 }
    }).collect();
    let foreground_mat = Mat::from_slice(&foreground)?;
    let foreground_mat = foreground_mat.reshape(1, roi.height)?;
    let kernel = opencv::imgproc::get_structuring_element(opencv::imgproc::MORPH_RECT,
        Size::new(3, 3), Point::new(-1, -1))?;
    let mut closed = Mat::default();
    opencv::imgproc::morphology_ex(&foreground_mat, &mut closed, opencv::imgproc::MORPH_CLOSE,
        &kernel, Point::new(-1, -1), 1, BORDER_DEFAULT, Scalar::default())?;
    let mut candidates: Vector<Vector<Point>> = Vector::new();
    opencv::imgproc::find_contours(&closed, &mut candidates, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point::new(x, y))?;
    let mut best = None;
    let mut best_area = 0.0;
    for candidate in candidates {
        let area = opencv::imgproc::contour_area(&candidate, false)?;
        if area > best_area {
            best_area = area;
            best = Some(candidate);
        }
    }
    let original_area = calculate_contour_area(&contour.points);
    if best_area < original_area * 0.5 || best_area > original_area * 4.0 {
        println!("Foreground refinement was inconclusive; retaining the edge outline");
        return Ok(contour.clone());
    }
    match best {
        Some(points) => Ok(Contour {
            points: points.iter().map(|p| Point2DMm::new(p.x as f64, p.y as f64)).collect(),
            closed: true,
        }),
        None => Ok(contour.clone()),
    }
}

/// Metrics for evaluating contour quality
#[derive(Clone, Copy)]
struct ContourMetrics {
    area: f64,
    perimeter: f64,
    compactness: f64,  // Circularity measure: 4π * area / perimeter²
    centroid_x: f64,
    centroid_y: f64,
    bbox: (f64, f64, f64, f64),  // min_x, min_y, max_x, max_y
}

/// Calculate comprehensive metrics for a contour
fn calculate_contour_metrics(points: &[Point2DMm], img_width: f64, img_height: f64) -> ContourMetrics {
    if points.len() < 3 {
        return ContourMetrics {
            area: 0.0,
            perimeter: 0.0,
            compactness: 0.0,
            centroid_x: img_width / 2.0,
            centroid_y: img_height / 2.0,
            bbox: (0.0, 0.0, 0.0, 0.0),
        };
    }

    // Calculate area using shoelace formula
    let area = calculate_contour_area(points);

    // Calculate perimeter
    let mut perimeter = 0.0;
    let n = points.len();
    for i in 0..n {
        let j = (i + 1) % n;
        let dx = points[j].x - points[i].x;
        let dy = points[j].y - points[i].y;
        perimeter += (dx * dx + dy * dy).sqrt();
    }

    // Calculate compactness (circularity): 4π * area / perimeter²
    // Perfect circle = 1.0, elongated shapes < 1.0
    let compactness = if perimeter > 0.0 {
        (4.0 * std::f64::consts::PI * area) / (perimeter * perimeter)
    } else {
        0.0
    };

    // Calculate centroid
    let mut cx = 0.0;
    let mut cy = 0.0;
    for point in points {
        cx += point.x;
        cy += point.y;
    }
    cx /= points.len() as f64;
    cy /= points.len() as f64;

    // Calculate bounding box
    let min_x = points.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
    let min_y = points.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
    let max_x = points.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max);
    let max_y = points.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max);

    ContourMetrics {
        area,
        perimeter,
        compactness,
        centroid_x: cx,
        centroid_y: cy,
        bbox: (min_x, min_y, max_x, max_y),
    }
}

/// Score based on area relative to image size
/// Prefer moderate-sized objects (not too small, not too large like full paper)
fn calculate_area_score(area: f64, image_area: f64) -> f64 {
    let ratio = area / image_area;

    // Ideal object size: 5-40% of image
    // Score peaks around 15-20% of image area
    if ratio < 0.001 {
        0.0  // Too small (noise)
    } else if ratio < 0.05 {
        ratio / 0.05  // Linearly increase from 0 to 1
    } else if ratio <= 0.40 {
        1.0  // Ideal range
    } else if ratio < 0.70 {
        1.0 - (ratio - 0.40) / 0.30  // Penalize large areas (likely paper/background)
    } else {
        0.1  // Heavily penalize very large areas (definitely background)
    }
}

/// Score based on distance from image center
/// Prefer centered objects over edge objects
fn calculate_centrality_score(
    cx: f64,
    cy: f64,
    img_cx: f64,
    img_cy: f64,
    img_width: f64,
    img_height: f64,
) -> f64 {
    let dx = (cx - img_cx).abs();
    let dy = (cy - img_cy).abs();
    let max_dist = ((img_width / 2.0).powi(2) + (img_height / 2.0).powi(2)).sqrt();
    let dist = (dx * dx + dy * dy).sqrt();

    // Score decreases with distance from center
    (1.0 - (dist / max_dist)).max(0.0)
}

/// Score based on proximity to image edges
/// Penalize contours very close to edges (likely paper boundaries)
fn calculate_edge_proximity_score(
    bbox: &(f64, f64, f64, f64),
    img_width: f64,
    img_height: f64,
) -> f64 {
    let (min_x, min_y, max_x, max_y) = *bbox;

    // Calculate minimum distance to any edge
    let dist_left = min_x;
    let dist_top = min_y;
    let dist_right = img_width - max_x;
    let dist_bottom = img_height - max_y;

    let min_edge_dist = dist_left.min(dist_top).min(dist_right).min(dist_bottom);

    // If contour is within 5% of image dimension from edge, heavily penalize
    let edge_threshold = (img_width.min(img_height)) * 0.05;

    if min_edge_dist < edge_threshold {
        0.0  // Very close to edge, likely paper boundary
    } else if min_edge_dist < edge_threshold * 3.0 {
        min_edge_dist / (edge_threshold * 3.0)  // Gradually increase score
    } else {
        1.0  // Far from edges, good
    }
}

/// Calculate the area of a contour using the shoelace formula
fn calculate_contour_area(points: &[Point2DMm]) -> f64 {
    if points.len() < 3 {
        return 0.0;
    }

    let mut area = 0.0;
    let n = points.len();

    for i in 0..n {
        let j = (i + 1) % n;
        area += points[i].x * points[j].y;
        area -= points[j].x * points[i].y;
    }

    (area / 2.0).abs()
}

/// Save debug image showing all contours with the best one highlighted
fn save_best_contour_debug(
    contours: &[Contour],
    best_idx: usize,
    source_image: &RgbImage,
    path: &str,
) -> Result<()> {
    use opencv::core::{Mat, Point, Scalar, Vector};
    use opencv::imgcodecs;

    // Convert source image to OpenCV Mat
    let img_data: Vec<u8> = source_image.as_raw().clone();
    let mat = Mat::from_slice(&img_data)?;
    let reshaped = mat.reshape(3, source_image.height() as i32)?;
    let mut debug_img = reshaped.try_clone()?;

    // Draw all contours in gray
    for (idx, contour) in contours.iter().enumerate() {
        let color = if idx == best_idx {
            Scalar::new(0.0, 255.0, 0.0, 0.0) // Green for best
        } else {
            Scalar::new(128.0, 128.0, 128.0, 0.0) // Gray for others
        };

        let thickness = if idx == best_idx { 3 } else { 1 };

        // Convert to OpenCV format
        let mut opencv_contour: Vector<Point> = Vector::new();
        for pt in &contour.points {
            opencv_contour.push(Point::new(pt.x as i32, pt.y as i32));
        }

        let mut contour_vec: Vector<Vector<Point>> = Vector::new();
        contour_vec.push(opencv_contour);

        opencv::imgproc::draw_contours(
            &mut debug_img,
            &contour_vec,
            0,
            color,
            thickness,
            opencv::imgproc::LINE_8,
            &Mat::default(),
            i32::MAX,
            Point::new(0, 0),
        )?;
    }

    imgcodecs::imwrite(path, &debug_img, &Vector::new())?;
    println!("Saved best contour debug image to: {}", path);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rectangle(x: f64, y: f64, width: f64, height: f64) -> Contour {
        Contour {
            points: vec![
                Point2DMm::new(x, y), Point2DMm::new(x + width, y),
                Point2DMm::new(x + width, y + height), Point2DMm::new(x, y + height),
            ],
            closed: true,
        }
    }

    #[test]
    fn selects_elongated_tool_over_small_compact_scale_marks() {
        let img = RgbImage::new(1000, 1000);
        let tool = rectangle(480.0, 100.0, 40.0, 800.0);
        assert!(calculate_contour_metrics(&tool.points, 1000.0, 1000.0).compactness < 0.15);
        let selected = filter_largest_contour(vec![
            rectangle(150.0, 500.0, 10.0, 10.0), tool.clone(),
            rectangle(0.0, 0.0, 999.0, 999.0),
        ], None, &img).unwrap();
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].points, tool.points);
    }

    #[test]
    fn no_valid_candidate_returns_empty_without_panicking() {
        let img = RgbImage::new(1000, 1000);
        let mut invalid = rectangle(0.0, 0.0, 10.0, 10.0);
        invalid.points[0].x = f64::NAN;
        assert!(filter_largest_contour(vec![
            invalid, rectangle(20.0, 20.0, 0.0, 100.0),
            rectangle(0.0, 0.0, 999.0, 999.0),
        ], None, &img).unwrap().is_empty());
    }

    #[test]
    fn foreground_refinement_recovers_missing_outline_without_filling_concavities() {
        let mut img = RgbImage::from_pixel(200, 300, image::Rgb([240; 3]));
        for (x, y, w, h) in [(80, 50, 40, 210), (60, 30, 80, 40), (10, 270, 5, 5)] {
            for py in y..y + h {
                for px in x..x + w {
                    img.put_pixel(px, py, image::Rgb([60, 70, 80]));
                }
            }
        }
        let refined = refine_edge_contours(vec![rectangle(85.0, 50.0, 30.0, 205.0)],
            &img, &[], None).unwrap();
        assert_eq!(refined.len(), 1);
        let metrics = calculate_contour_metrics(&refined[0].points, 200.0, 300.0);
        for (actual, expected) in [
            (metrics.bbox.0, 60.0), (metrics.bbox.1, 30.0),
            (metrics.bbox.2, 139.0), (metrics.bbox.3, 259.0),
        ] {
            assert!((actual - expected).abs() <= 1.0);
        }
        let polygon: Vector<Point> = refined[0].points.iter()
            .map(|p| Point::new(p.x as i32, p.y as i32)).collect();
        assert!(opencv::imgproc::point_polygon_test(&polygon,
            opencv::core::Point2f::new(65.0, 50.0), false).unwrap() > 0.0);
        assert!(opencv::imgproc::point_polygon_test(&polygon,
            opencv::core::Point2f::new(70.0, 100.0), false).unwrap() < 0.0);
    }
}
