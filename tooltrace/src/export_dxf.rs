// DXF export module
// Exports traced contours to DXF format (AutoCAD/Fusion 360 compatible)

use anyhow::Result;
use dxf::{Drawing, entities::*, Color, LwPolylineVertex};
use dxf::enums::AcadVersion;
use tooltrace_common::Contour;

/// Export contours to DXF file
/// Coordinates are in millimeters
pub fn export_dxf(contours: &[Contour], output_path: &str) -> Result<()> {
    let drawing = build_drawing(contours);
    drawing.save_file(output_path)?;
    let (min_x, min_y, max_x, max_y) = calculate_bounds(contours);
    println!("Exported {} contour(s) to DXF: {}", contours.len(), output_path);
    println!("  Bounds: {:.1}mm × {:.1}mm", max_x - min_x, max_y - min_y);
    println!("  Units: millimeters");
    Ok(())
}

fn build_drawing(contours: &[Contour]) -> Drawing {
    let mut drawing = Drawing::new();
    drawing.header.version = AcadVersion::R2010;

    // Images/SVG use Y down; CAD uses Y up. Reflect all contours about one
    // shared horizontal center so their placement and dimensions are retained.
    let (_, min_y, _, max_y) = calculate_bounds(contours);
    let y_reflection = min_y + max_y;

    // Export each contour as a polyline
    for (idx, contour) in contours.iter().enumerate() {
        if contour.points.is_empty() {
            continue;
        }

        // Create a LWPOLYLINE (lightweight polyline) for 2D contours
        let mut polyline = LwPolyline::default();

        // Add vertices
        for pt in &contour.points {
            polyline.vertices.push(LwPolylineVertex {
                x: pt.x,
                y: y_reflection - pt.y,
                .. Default::default()
            });
        }

        // Close the polyline if contour is closed
        polyline.set_is_closed(contour.closed);

        // Create entity with common properties
        let mut common = EntityCommon::default();
        common.layer = format!("CONTOUR-{}", idx);
        common.color = Color::from_index(7); // White/Black (default)

        let entity = Entity {
            common,
            specific: EntityType::LwPolyline(polyline),
        };

        // Add entity to drawing
        drawing.add_entity(entity);
    }

    drawing
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use tooltrace_common::Point2DMm;

    fn round_trip(contours: &[Contour]) -> Vec<LwPolyline> {
        let mut bytes = Vec::new();
        build_drawing(contours).save(&mut bytes).unwrap();
        let drawing = Drawing::load(&mut Cursor::new(bytes)).unwrap();
        drawing.entities().filter_map(|entity| {
            if let EntityType::LwPolyline(polyline) = &entity.specific { Some(polyline.clone()) }
            else { None }
        }).collect()
    }

    #[test]
    fn asymmetric_shape_keeps_its_visual_orientation_in_cad() {
        let contour = Contour {
            points: [(10.0,20.0),(40.0,20.0),(40.0,30.0),(20.0,30.0),(20.0,70.0),(10.0,70.0)]
                .into_iter().map(|(x,y)| Point2DMm::new(x,y)).collect(),
            closed: true,
        };
        let polylines = round_trip(&[contour]);
        assert_eq!(polylines.len(), 1);
        assert!(polylines[0].is_closed());
        let expected = [(10.0,70.0),(40.0,70.0),(40.0,60.0),(20.0,60.0),(20.0,20.0),(10.0,20.0)];
        for (vertex, (x,y)) in polylines[0].vertices.iter().zip(expected) {
            assert_eq!((vertex.x,vertex.y),(x,y));
        }
    }

    #[test]
    fn multiple_contours_share_the_same_coordinate_transform() {
        let contours = [
            Contour { points: vec![Point2DMm::new(10.0,10.0),Point2DMm::new(40.0,10.0),
                Point2DMm::new(40.0,110.0),Point2DMm::new(10.0,110.0)], closed: true },
            Contour { points: vec![Point2DMm::new(20.0,20.0),Point2DMm::new(30.0,20.0),
                Point2DMm::new(30.0,30.0),Point2DMm::new(20.0,30.0)], closed: true },
        ];
        let polylines = round_trip(&contours);
        assert_eq!(polylines[0].vertices[0].y,110.0);
        // The small contour was near the image top, so it must be near the CAD top.
        assert_eq!(polylines[1].vertices[0].y,100.0);
        assert_eq!(polylines[1].vertices[2].y,90.0);
    }
}

/// Calculate bounding box for all contours
fn calculate_bounds(contours: &[Contour]) -> (f64, f64, f64, f64) {
    let mut min_x = f64::MAX;
    let mut min_y = f64::MAX;
    let mut max_x = f64::MIN;
    let mut max_y = f64::MIN;

    for contour in contours {
        for pt in &contour.points {
            min_x = min_x.min(pt.x);
            min_y = min_y.min(pt.y);
            max_x = max_x.max(pt.x);
            max_y = max_y.max(pt.y);
        }
    }

    if min_x < f64::MAX {
        (min_x, min_y, max_x, max_y)
    } else {
        (0.0, 0.0, 100.0, 100.0) // Default if no points
    }
}
