use clap::Parser;
use anyhow::Result;
use tooltrace_common::{OutputFormat, DEFAULT_TAG_SIZE_MM};

mod detection;
mod calibration;
mod calibration_fit;
mod measurement;
mod segmentation;
mod fastsam;
mod attachments;
mod tracing;
mod export_svg;
mod export_dxf;
mod smoothing;
mod geometry;

/// Analyze photographs of objects on calibration paper and generate vector traces
#[derive(Parser, Debug)]
#[command(name = "tooltrace")]
#[command(about = "Trace objects from calibration paper photos", long_about = None)]
struct Args {
    /// Input image file path
    #[arg(short, long)]
    input: String,

    /// Output file path (without extension)
    #[arg(short, long, default_value = "output")]
    output: String,

    /// Output format
    #[arg(short, long, value_enum, default_value = "both")]
    format: FormatArg,

    /// Enable debug mode (save intermediate visualizations)
    #[arg(short, long)]
    debug: bool,

    /// AprilTag size in millimeters (outer black square)
    #[arg(short, long, default_value_t = DEFAULT_TAG_SIZE_MM)]
    tag_size: f64,

    /// Limit the longest image side for tag detection (default: native resolution)
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
    detection_max_dimension: Option<u32>,

    /// Retry incomplete tag detection with 1 px Gaussian smoothing
    #[arg(long)]
    detection_smoothing: bool,

    /// FastSAM ONNX model (default: automatically search models/ near the executable)
    #[arg(long)]
    fastsam_model: Option<std::path::PathBuf>,

    /// Disable recovery of small FastSAM parts attached to the main object
    #[arg(long)]
    no_attachments: bool,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum FormatArg {
    Svg,
    Dxf,
    Both,
}

impl From<FormatArg> for OutputFormat {
    fn from(arg: FormatArg) -> Self {
        match arg {
            FormatArg::Svg => OutputFormat::Svg,
            FormatArg::Dxf => OutputFormat::Dxf,
            FormatArg::Both => OutputFormat::Both,
        }
    }
}

#[cfg(test)]
mod cli_tests {
    use super::*;

    #[test]
    fn detection_preprocessing_is_opt_in() {
        let args = Args::try_parse_from(["tooltrace", "--input", "photo.jpg"]).unwrap();
        assert_eq!(args.detection_max_dimension, None);
        assert!(!args.detection_smoothing);
        let args = Args::try_parse_from([
            "tooltrace", "--input", "photo.jpg", "--detection-max-dimension", "1920",
            "--detection-smoothing",
        ]).unwrap();
        assert_eq!(args.detection_max_dimension, Some(1920));
        assert!(args.detection_smoothing);
        assert!(Args::try_parse_from([
            "tooltrace", "--input", "photo.jpg", "--detection-max-dimension", "0",
        ]).is_err());
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("ToolTrace - Object Tracing Tool");
    println!("================================");
    println!("Input: {}", args.input);
    println!("Output: {}", args.output);
    println!("Format: {:?}", args.format);
    println!("Tag size: {}mm", args.tag_size);
    println!("Debug mode: {}", args.debug);
    println!();

    // Step 1: Detect AprilTags
    println!("Step 1: Detecting AprilTags...");
    let debug_path = if args.debug {
        Some(format!("{}_debug_detection.jpg", args.output))
    } else {
        None
    };
    let detection_options = detection::DetectionOptions {
        max_dimension: args.detection_max_dimension,
        smoothing: args.detection_smoothing,
    };
    let detections = detection::detect_apriltags(&args.input, detection_options, debug_path.as_deref())?;

    if detections.is_empty() {
        anyhow::bail!("No AprilTags detected in the image. Make sure the calibration paper is visible.");
    }

    println!("\nDetected {} tag(s)", detections.len());
    for det in &detections {
        println!("  - ID {}: center at ({:.1}, {:.1})", det.id, det.center.0, det.center.1);
    }

    // Step 2: Calculate perspective transform
    println!("\nStep 2: Calculating calibration and perspective transform...");
    let input_img = image::open(&args.input)?.to_rgb8();
    let debug_prefix = if args.debug { Some(args.output.as_str()) } else { None };
    let calibration = calibration::calculate_calibration(&detections, args.tag_size, &input_img, debug_prefix)?;

    // Save rotated and cropped image if debug mode
    if args.debug {
        if let Some(ref cropped) = calibration.rotated_image {
            let cropped_path = format!("{}_cropped.jpg", args.output);
            cropped.save(&cropped_path)?;
            println!("Saved cropped image to: {}", cropped_path);
        }
    }

    // Step 3: Apply perspective correction to flatten the paper
    println!("\nStep 3: Applying perspective correction...");
    let source_for_correction = calibration.rotated_image.as_ref().unwrap_or(&input_img);
    let flattened = calibration::apply_perspective_correction(source_for_correction, &calibration)?;

    // Save the flattened image
    let flattened_path = format!("{}_flattened.jpg", args.output);
    flattened.save(&flattened_path)?;
    println!("Saved flattened image to: {}", flattened_path);

    let axis_checks = measurement::check_measurement_axes(
        &flattened, calibration.paper_size, args.tag_size,
    )?;
    let axis_report = measurement::report(&axis_checks);
    println!("\n{axis_report}");
    let fit_status = if calibration.fit_max_mm > measurement::TOLERANCE_MM {
        "Remaining marker mismatch exceeds 0.50 mm; a ruler result within tolerance does not validate the whole image.\n"
    } else {
        "Marker fitting residuals are within 0.50 mm; these are not independent validation.\n"
    };
    println!("{fit_status}");
    let calibration_report = format!(
        "Input: {}\n{}\n{}\n{}",
        args.input, calibration.fit_report, fit_status, axis_report);
    std::fs::write(format!("{}_calibration.txt", args.output), calibration_report)?;
    if args.debug {
        measurement::save_overlay(
            &flattened, &axis_checks, &format!("{}_measurement_axes.jpg", args.output),
        )?;
    }

    // Step 4: Segment objects from the flattened image
    println!("\nStep 4: Segmenting objects...");

    // Calculate AprilTag exclusion regions in flattened image space (at 300 DPI)
    let dpi = 300.0;
    let pixels_per_mm = dpi / 25.4; // 11.811 pixels/mm
    let margin_mm = 15.0; // Tags are 15mm from paper edge
    let tag_size_mm = args.tag_size;

    // Calculate tag regions: [top-left, top-right, bottom-right, bottom-left]
    let tag_regions: Vec<(i32, i32, i32, i32)> = vec![
        // Top-left tag
        (
            (margin_mm * pixels_per_mm) as i32,
            (margin_mm * pixels_per_mm) as i32,
            (tag_size_mm * pixels_per_mm) as i32,
            (tag_size_mm * pixels_per_mm) as i32,
        ),
        // Top-right tag
        (
            (calibration.output_width as f64 - (margin_mm + tag_size_mm) * pixels_per_mm) as i32,
            (margin_mm * pixels_per_mm) as i32,
            (tag_size_mm * pixels_per_mm) as i32,
            (tag_size_mm * pixels_per_mm) as i32,
        ),
        // Bottom-right tag
        (
            (calibration.output_width as f64 - (margin_mm + tag_size_mm) * pixels_per_mm) as i32,
            (calibration.output_height as f64 - (margin_mm + tag_size_mm) * pixels_per_mm) as i32,
            (tag_size_mm * pixels_per_mm) as i32,
            (tag_size_mm * pixels_per_mm) as i32,
        ),
        // Bottom-left tag
        (
            (margin_mm * pixels_per_mm) as i32,
            (calibration.output_height as f64 - (margin_mm + tag_size_mm) * pixels_per_mm) as i32,
            (tag_size_mm * pixels_per_mm) as i32,
            (tag_size_mm * pixels_per_mm) as i32,
        ),
    ];

    println!("Calculated {} AprilTag exclusion regions", tag_regions.len());

    let debug_contours_path = if args.debug {
        Some(format!("{}_contours.jpg", args.output))
    } else {
        None
    };
    let segmented = segmentation::segment_object(&flattened, &tag_regions, args.fastsam_model.as_deref(), debug_contours_path.as_deref())?;
    let contours_pixels = segmented.contours;

    if contours_pixels.is_empty() {
        println!("\nNo contours detected. Try adjusting the object placement or lighting.");
        return Ok(());
    }

    println!("Found {} contour(s) in total", contours_pixels.len());
    let part_candidates = if !segmented.edge_based && !args.no_attachments {
        contours_pixels.clone()
    } else { Vec::new() };

    // Step 5a: Export multi-color SVG with all contours (debug visualization)
    if args.debug && contours_pixels.len() > 1 {
        println!("\nStep 5a: Exporting multi-color debug SVG with all contours...");
        let dpi = 300.0;
        let all_contours_mm = segmentation::pixels_to_mm(contours_pixels.clone(), dpi);
        let multicolor_svg_path = format!("{}_all_contours_multicolor.svg", args.output);
        export_svg::export_svg_multi_color(&all_contours_mm, &multicolor_svg_path)?;
    }

    // Step 5b: Filter to largest contour only
    println!("\nStep 5b: Filtering to largest contour...");
    let largest_debug_path = if args.debug {
        Some(format!("{}_largest_contour.jpg", args.output))
    } else {
        None
    };
    let mut largest_contour_pixels = segmentation::filter_largest_contour(
        contours_pixels,
        largest_debug_path.as_deref(),
        &flattened,
    )?;

    if largest_contour_pixels.is_empty() {
        println!("\nNo valid contour after filtering.");
        return Ok(());
    }
    let original_primary = if part_candidates.is_empty() { Vec::new() } else { largest_contour_pixels.clone() };

    if segmented.edge_based {
        println!("\nRefining the foreground outline...");
        let refined_debug_path = if args.debug { Some(format!("{}_refined_contour.jpg", args.output)) } else { None };
        largest_contour_pixels = segmentation::refine_edge_contours(
            largest_contour_pixels, &flattened, &tag_regions, refined_debug_path.as_deref(),
        )?;
    }

    println!("\nAligning and straightening supported object edges...");
    let linear_debug_path = if args.debug { Some(format!("{}_linearized_contour.jpg", args.output)) } else { None };
    largest_contour_pixels = geometry::regularize_contours(
        largest_contour_pixels, &flattened, linear_debug_path.as_deref(),
    )?;

    if !part_candidates.is_empty() {
        println!("\nRecovering small attached FastSAM parts...");
        let attachment_debug = if args.debug { Some(format!("{}_attachments.jpg", args.output)) } else { None };
        largest_contour_pixels = attachments::complete_attachments(
            largest_contour_pixels, &original_primary, &part_candidates, &flattened,
            &tag_regions, attachment_debug.as_deref(),
        )?;
    }

    // Step 5c: Apply spline smoothing to create gentle low-frequency curves
    println!("\nStep 5c: Applying spline smoothing...");
    let control_spacing_pixels = 5; // Maximum distance between spline control points
    let smoothing_debug_path = if args.debug {
        Some(format!("{}_smoothed.jpg", args.output))
    } else {
        None
    };
    let smoothed_contours = smoothing::apply_spline_smoothing(
        largest_contour_pixels,
        control_spacing_pixels,
        smoothing_debug_path.as_deref(),
        Some(&flattened),
    )?;

    // Step 6: Convert pixel coordinates to millimeters
    println!("\nStep 6: Converting coordinates to millimeters...");
    let dpi = 300.0; // Flattened image is at 300 DPI
    let contours_mm = segmentation::pixels_to_mm(smoothed_contours, dpi);

    println!("Final result: {} smoothed contour(s) in millimeter coordinates", contours_mm.len());

    // Step 7: Export to vector format(s)
    println!("\nStep 7: Exporting to vector format(s)...");
    let format = OutputFormat::from(args.format);

    match format {
        OutputFormat::Svg => {
            let svg_path = format!("{}.svg", args.output);
            export_svg::export_svg(&contours_mm, &svg_path)?;
        }
        OutputFormat::Dxf => {
            let dxf_path = format!("{}.dxf", args.output);
            export_dxf::export_dxf(&contours_mm, &dxf_path)?;
        }
        OutputFormat::Both => {
            let svg_path = format!("{}.svg", args.output);
            let dxf_path = format!("{}.dxf", args.output);
            export_svg::export_svg(&contours_mm, &svg_path)?;
            export_dxf::export_dxf(&contours_mm, &dxf_path)?;
        }
    }

    println!("\n✓ Processing complete!");

    Ok(())
}
