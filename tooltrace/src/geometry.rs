//! Match the silhouette to sharp image edges and regularize long straight runs.
use anyhow::Result;
use image::{GrayImage, RgbImage};
use opencv::{
    core::{Mat, Point, Point2f, Scalar, Vec4f, Vector, CV_8UC1},
    imgproc,
    prelude::*,
};
use tooltrace_common::{Contour, Point2DMm};

const PIXELS_PER_MM: f64 = 300.0 / 25.4;
const MIN_LINE_LENGTH: f64 = 8.0 * PIXELS_PER_MM;
const LINE_TOLERANCE: f64 = 0.6 * PIXELS_PER_MM;

#[derive(Clone, Copy)]
struct Line {
    a: Point2DMm,
    b: Point2DMm,
}

impl Line {
    fn length(self) -> f64 {
        distance(self.a, self.b)
    }
    fn direction(self) -> (f64, f64) {
        let length = self.length();
        (
            (self.b.x - self.a.x) / length,
            (self.b.y - self.a.y) / length,
        )
    }
    fn project(self, p: Point2DMm) -> (Point2DMm, f64) {
        let (dx, dy) = self.direction();
        let t = (p.x - self.a.x) * dx + (p.y - self.a.y) * dy;
        (Point2DMm::new(self.a.x + t * dx, self.a.y + t * dy), t)
    }
}

fn distance(a: Point2DMm, b: Point2DMm) -> f64 {
    (a.x - b.x).hypot(a.y - b.y)
}

fn gray_at(gray: &GrayImage, p: Point2DMm) -> f64 {
    let x = p.x.clamp(0.0, (gray.width() - 1) as f64);
    let y = p.y.clamp(0.0, (gray.height() - 1) as f64);
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(gray.width() - 1);
    let y1 = (y0 + 1).min(gray.height() - 1);
    let (fx, fy) = (x - x0 as f64, y - y0 as f64);
    (1.0 - fy)
        * ((1.0 - fx) * gray.get_pixel(x0, y0)[0] as f64 + fx * gray.get_pixel(x1, y0)[0] as f64)
        + fy * ((1.0 - fx) * gray.get_pixel(x0, y1)[0] as f64
            + fx * gray.get_pixel(x1, y1)[0] as f64)
}

fn sharpness(line: Line, gray: &GrayImage) -> f64 {
    let (dx, dy) = line.direction();
    let samples = (line.length() / 10.0).ceil().max(3.0) as usize;
    (0..samples)
        .map(|i| {
            let t = (i as f64 + 0.5) / samples as f64;
            let p = Point2DMm::new(
                line.a.x + t * (line.b.x - line.a.x),
                line.a.y + t * (line.b.y - line.a.y),
            );
            (gray_at(gray, Point2DMm::new(p.x - 2.0 * dy, p.y + 2.0 * dx))
                - gray_at(gray, Point2DMm::new(p.x + 2.0 * dy, p.y - 2.0 * dx)))
            .abs()
                / 4.0
        })
        .sum::<f64>()
        / samples as f64
}

fn evidence_lines(gray: &GrayImage) -> Result<Vec<Line>> {
    let mat = Mat::from_slice(gray.as_raw())?;
    let mat = mat.reshape(1, gray.height() as i32)?;
    // Explicit scale 1.0: the edge evidence comes from native pixels.
    let mut detector = imgproc::create_line_segment_detector(
        imgproc::LSD_REFINE_STD,
        1.0,
        0.6,
        1.0,
        22.5,
        0.0,
        0.7,
        1024,
    )?;
    let mut detected: Vector<Vec4f> = Vector::new();
    detector.detect_def(&mat, &mut detected)?;
    let mut lines: Vec<Line> = detected
        .iter()
        .map(|v| Line {
            a: Point2DMm::new(v[0] as f64, v[1] as f64),
            b: Point2DMm::new(v[2] as f64, v[3] as f64),
        })
        .filter(|line| line.length() >= 25.0 && sharpness(*line, gray) >= 2.0)
        .collect();
    lines.sort_by(|a, b| b.length().total_cmp(&a.length()));
    let mut merged: Vec<Line> = Vec::new();
    for line in lines {
        let (dx, dy) = line.direction();
        let mut matched = false;
        for existing in &mut merged {
            let (ex, ey) = existing.direction();
            if (dx * ex + dy * ey).abs() < 1.5_f64.to_radians().cos() {
                continue;
            }
            let (pa, ta) = existing.project(line.a);
            let (pb, tb) = existing.project(line.b);
            if distance(pa, line.a).max(distance(pb, line.b)) > 3.0 {
                continue;
            }
            let low = ta.min(tb);
            let high = ta.max(tb);
            if low > existing.length() + 60.0 || high < -60.0 {
                continue;
            }
            let origin = existing.a;
            let end = high.max(existing.length());
            let start = low.min(0.0);
            existing.a = Point2DMm::new(origin.x + start * ex, origin.y + start * ey);
            existing.b = Point2DMm::new(origin.x + end * ex, origin.y + end * ey);
            matched = true;
            break;
        }
        if !matched {
            merged.push(line);
        }
    }
    // A newly extended segment can bridge two groups that did not overlap when
    // their first fragments were seen. Finish those transitive merges as well.
    loop {
        let mut joined = false;
        'pairs: for i in 0..merged.len() {
            for j in i + 1..merged.len() {
                let a = merged[i];
                let b = merged[j];
                let (dx, dy) = a.direction();
                let (bx, by) = b.direction();
                if (dx * bx + dy * by).abs() < 1.5_f64.to_radians().cos() {
                    continue;
                }
                let (pa, ta) = a.project(b.a);
                let (pb, tb) = a.project(b.b);
                if distance(pa, b.a).max(distance(pb, b.b)) > 3.0 {
                    continue;
                }
                if ta.min(tb) > a.length() + 60.0 || ta.max(tb) < -60.0 {
                    continue;
                }
                let low = ta.min(tb).min(0.0);
                let high = ta.max(tb).max(a.length());
                merged[i] = Line {
                    a: Point2DMm::new(a.a.x + low * dx, a.a.y + low * dy),
                    b: Point2DMm::new(a.a.x + high * dx, a.a.y + high * dy),
                };
                merged.swap_remove(j);
                joined = true;
                break 'pairs;
            }
        }
        if !joined {
            break;
        }
    }
    merged.retain(|line| line.length() >= MIN_LINE_LENGTH);
    Ok(merged)
}

fn resample(points: &[Point2DMm], closed: bool) -> Vec<Point2DMm> {
    let mut result = Vec::new();
    let spans = if closed {
        points.len()
    } else {
        points.len().saturating_sub(1)
    };
    for i in 0..spans {
        let a = points[i];
        let b = points[(i + 1) % points.len()];
        let steps = (distance(a, b) / 3.0).ceil().max(1.0) as usize;
        for j in 0..steps {
            let t = j as f64 / steps as f64;
            result.push(Point2DMm::new(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y)));
        }
    }
    if !closed {
        if let Some(last) = points.last() {
            result.push(*last);
        }
    }
    result
}

fn snap_supported_runs(
    contour: &Contour,
    lines: &[Line],
    gray: &GrayImage,
) -> Result<Vec<Point2DMm>> {
    let (width, height) = gray.dimensions();
    let points = resample(&contour.points, contour.closed);
    if points.len() < 10 || !contour.closed {
        return Ok(points);
    }
    let polygon: Vector<Point> = contour
        .points
        .iter()
        .map(|p| Point::new(p.x.round() as i32, p.y.round() as i32))
        .collect();
    let polygons = Vector::<Vector<Point>>::from_iter([polygon]);
    let mut mask =
        Mat::new_rows_cols_with_default(height as i32, width as i32, CV_8UC1, Scalar::all(0.0))?;
    imgproc::draw_contours(
        &mut mask,
        &polygons,
        0,
        Scalar::all(255.0),
        -1,
        imgproc::LINE_8,
        &Mat::default(),
        i32::MAX,
        Point::new(0, 0),
    )?;
    let mask_data = mask.data_typed::<u8>()?;
    let signed_area: f64 = points
        .iter()
        .zip(points.iter().cycle().skip(1))
        .take(points.len())
        .map(|(a, b)| a.x * b.y - b.x * a.y)
        .sum();
    let orientation = if signed_area >= 0.0 { 1.0 } else { -1.0 };
    let mut assignments = vec![None; points.len()];
    for i in 0..points.len() {
        let p = points[i];
        let before = points[(i + points.len() - 4) % points.len()];
        let after = points[(i + 4) % points.len()];
        let span = distance(before, after);
        if span < 1.0 {
            continue;
        }
        let (tx, ty) = ((after.x - before.x) / span, (after.y - before.y) / span);
        let (nx, ny) = (-ty * orientation, tx * orientation);
        // Restrict inward searches to less than half the local object thickness.
        let mut thickness = 160.0;
        for d in 3..=160 {
            let (x, y) = (
                (p.x + nx * d as f64).round() as i32,
                (p.y + ny * d as f64).round() as i32,
            );
            if x < 0
                || y < 0
                || x >= width as i32
                || y >= height as i32
                || mask_data[y as usize * width as usize + x as usize] == 0
            {
                thickness = d as f64;
                break;
            }
        }
        let inward_limit = (6.0 * PIXELS_PER_MM).min(thickness * 0.45);
        let mut best = f64::INFINITY;
        let boundary_contrast = (gray_at(gray, Point2DMm::new(p.x + 2.0 * nx, p.y + 2.0 * ny))
            - gray_at(gray, Point2DMm::new(p.x - 2.0 * nx, p.y - 2.0 * ny)))
        .abs()
            / 4.0;
        for (id, line) in lines.iter().enumerate() {
            let (dx, dy) = line.direction();
            let angle_limit: f64 = if boundary_contrast < 2.5 { 45.0 } else { 15.0 };
            if (dx * tx + dy * ty).abs() < angle_limit.to_radians().cos() {
                continue;
            }
            let (projected, t) = line.project(p);
            if t < -60.0 || t > line.length() + 60.0 {
                continue;
            }
            let shift = (projected.x - p.x) * nx + (projected.y - p.y) * ny;
            if shift < -0.6 * PIXELS_PER_MM || shift > inward_limit {
                continue;
            }
            let error = distance(projected, p);
            if error < best {
                best = error;
                assignments[i] = Some(id);
            }
        }
    }
    // Do not leave little rectangular islands of soft shadow between two
    // supported portions of the same physical edge. Only bridge short, weak
    // boundary gaps; a sharp notch or a rounded protrusion remains a boundary.
    for (id, line) in lines.iter().enumerate() {
        let anchors: Vec<usize> = assignments
            .iter()
            .enumerate()
            .filter_map(|(i, &a)| if a == Some(id) { Some(i) } else { None })
            .collect();
        if anchors.len() < 2 {
            continue;
        }
        for k in 0..anchors.len() {
            let first = anchors[k];
            let last = if k + 1 < anchors.len() {
                anchors[k + 1]
            } else {
                anchors[0] + points.len()
            };
            if last <= first + 1 {
                continue;
            }
            let gap_length: f64 = (first + 1..=last)
                .map(|i| distance(points[(i - 1) % points.len()], points[i % points.len()]))
                .sum();
            if gap_length > 10.0 * PIXELS_PER_MM {
                continue;
            }
            let mut weak = 0;
            let mut near = true;
            for j in first + 1..last {
                let i = j % points.len();
                let p = points[i];
                if distance(p, line.project(p).0) > 6.0 * PIXELS_PER_MM {
                    near = false;
                    break;
                }
                let a = points[(i + points.len() - 4) % points.len()];
                let b = points[(i + 4) % points.len()];
                let length = distance(a, b).max(1.0);
                let (nx, ny) = (-(b.y - a.y) / length, (b.x - a.x) / length);
                let contrast = (gray_at(gray, Point2DMm::new(p.x + 2.0 * nx, p.y + 2.0 * ny))
                    - gray_at(gray, Point2DMm::new(p.x - 2.0 * nx, p.y - 2.0 * ny)))
                .abs()
                    / 4.0;
                if contrast < 6.0 {
                    weak += 1;
                }
            }
            let baseline = Line {
                a: points[first],
                b: points[last % points.len()],
            };
            let originally_straight = baseline.length() >= 1.0
                && (first + 1..last).all(|j| {
                    let p = points[j % points.len()];
                    distance(p, baseline.project(p).0) <= LINE_TOLERANCE
                });
            if near && (weak as f64 / (last - first - 1) as f64 >= 0.6 || originally_straight) {
                for j in first + 1..last {
                    assignments[j % points.len()] = Some(id);
                }
            }
        }
    }
    let mut result = points.clone();
    let start = (0..points.len())
        .find(|&i| assignments[i] != assignments[(i + points.len() - 1) % points.len()])
        .unwrap_or(0);
    let mut offset = 0;
    while offset < points.len() {
        let begin = offset;
        let id = assignments[(start + offset) % points.len()];
        while offset < points.len() && assignments[(start + offset) % points.len()] == id {
            offset += 1;
        }
        if let Some(id) = id {
            let length: f64 = (begin + 1..offset)
                .map(|j| {
                    distance(
                        points[(start + j - 1) % points.len()],
                        points[(start + j) % points.len()],
                    )
                })
                .sum();
            if length >= MIN_LINE_LENGTH {
                for j in begin..offset {
                    let i = (start + j) % points.len();
                    result[i] = lines[id].project(points[i]).0;
                }
            }
        }
    }
    Ok(result)
}

fn straighten_runs(points: &[Point2DMm], closed: bool) -> Result<Vec<Point2DMm>> {
    let input: Vector<Point2f> = points
        .iter()
        .map(|p| Point2f::new(p.x as f32, p.y as f32))
        .collect();
    let mut simplified: Vector<Point2f> = Vector::new();
    imgproc::approx_poly_dp(&input, &mut simplified, 1.0, closed)?;
    let mut vertices: Vec<Point2DMm> = simplified
        .iter()
        .map(|p| Point2DMm::new(p.x as f64, p.y as f64))
        .collect();
    if vertices.len() < 3 {
        return Ok(vertices);
    }
    if closed {
        let corner = (0..vertices.len())
            .filter_map(|i| {
                let a = vertices[(i + vertices.len() - 1) % vertices.len()];
                let b = vertices[i];
                let c = vertices[(i + 1) % vertices.len()];
                let l1 = distance(a, b);
                let l2 = distance(b, c);
                if l1 < 15.0 || l2 < 15.0 {
                    return None;
                }
                Some((
                    i,
                    ((b.x - a.x) * (c.x - b.x) + (b.y - a.y) * (c.y - b.y)) / (l1 * l2),
                ))
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|v| v.0)
            .unwrap_or(0);
        vertices.rotate_left(corner);
        vertices.push(vertices[0]);
    }
    let mut result = Vec::new();
    let mut i = 0;
    let limit = if closed {
        vertices.len() - 1
    } else {
        vertices.len()
    };
    while i < limit {
        result.push(vertices[i]);
        let mut end = i + 1;
        let mut candidate = end + 1;
        while candidate < vertices.len() {
            let line = Line {
                a: vertices[i],
                b: vertices[candidate],
            };
            if line.length() < 1.0 {
                break;
            }
            let max_error = vertices[i + 1..candidate]
                .iter()
                .map(|&p| distance(p, line.project(p).0))
                .fold(0.0, f64::max);
            if max_error > 3.0 * LINE_TOLERANCE {
                break;
            }
            if max_error <= LINE_TOLERANCE && line.length() >= MIN_LINE_LENGTH {
                // A coherent change of tangent indicates a real arc, even if
                // its shallow bow would fit within the position tolerance.
                let anchor = |reverse: bool| {
                    let mut index = if reverse { candidate } else { i };
                    let stop = if reverse { i } else { candidate };
                    let mut remaining = 40.0;
                    while index != stop {
                        let next = if reverse { index - 1 } else { index + 1 };
                        let a = vertices[index];
                        let b = vertices[next];
                        let length = distance(a, b);
                        if length >= remaining {
                            let t = remaining / length;
                            return Point2DMm::new(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y));
                        }
                        remaining -= length;
                        index = next;
                    }
                    vertices[stop]
                };
                let (ax, ay) = Line {
                    a: vertices[i],
                    b: anchor(false),
                }
                .direction();
                let (bx, by) = Line {
                    a: anchor(true),
                    b: vertices[candidate],
                }
                .direction();
                if ax * bx + ay * by >= 12.0_f64.to_radians().cos() {
                    end = candidate;
                }
            }
            candidate += 1;
        }
        if end < vertices.len() && distance(vertices[i], vertices[end]) >= MIN_LINE_LENGTH {
            i = end;
        } else {
            i += 1;
        }
    }
    Ok(result)
}

pub fn regularize_contours(
    contours: Vec<Contour>,
    source: &RgbImage,
    debug_path: Option<&str>,
) -> Result<Vec<Contour>> {
    let gray = image::imageops::grayscale(source);
    let lines = evidence_lines(&gray)?;
    println!(
        "Found {} supported long image edges for contour refinement",
        lines.len()
    );
    let mut result = Vec::new();
    for contour in contours {
        let aligned = snap_supported_runs(&contour, &lines, &gray)?;
        result.push(Contour {
            points: straighten_runs(&aligned, contour.closed)?,
            closed: contour.closed,
        });
    }
    if let Some(path) = debug_path {
        let data = Mat::from_slice(source.as_raw())?;
        let data = data.reshape(3, source.height() as i32)?;
        let mut preview = Mat::default();
        imgproc::cvt_color(
            &data,
            &mut preview,
            imgproc::COLOR_RGB2BGR,
            0,
            opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;
        for contour in &result {
            let points: Vector<Point> = contour
                .points
                .iter()
                .map(|p| Point::new(p.x.round() as i32, p.y.round() as i32))
                .collect();
            let contours = Vector::<Vector<Point>>::from_iter([points]);
            imgproc::draw_contours(
                &mut preview,
                &contours,
                0,
                Scalar::new(0.0, 255.0, 0.0, 0.0),
                2,
                imgproc::LINE_8,
                &Mat::default(),
                i32::MAX,
                Point::new(0, 0),
            )?;
        }
        opencv::imgcodecs::imwrite(path, &preview, &Vector::new())?;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noisy_long_straight_edge_becomes_one_segment() {
        let points: Vec<_> = (0..=100)
            .map(|i| {
                let x = i as f64 * 4.0;
                Point2DMm::new(x, 50.0 + 0.2 * x + if i % 2 == 0 { 1.5 } else { -1.5 })
            })
            .collect();
        let result = straighten_runs(&points, false).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], points[0]);
        assert_eq!(*result.last().unwrap(), *points.last().unwrap());
    }

    #[test]
    fn rounded_contours_do_not_turn_into_long_flat_facets() {
        for radius in [80.0, 200.0] {
            let points: Vec<_> = (0..720)
                .map(|i| {
                    let a = i as f64 * std::f64::consts::TAU / 720.0;
                    Point2DMm::new(250.0 + radius * a.cos(), 250.0 + radius * a.sin())
                })
                .collect();
            let result = straighten_runs(&points, true).unwrap();
            assert!(result.len() >= 16);
            for i in 0..result.len() {
                let a = result[i];
                let b = result[(i + 1) % result.len()];
                let middle = Point2DMm::new((a.x + b.x) / 2.0, (a.y + b.y) / 2.0);
                assert!((distance(middle, Point2DMm::new(250.0, 250.0)) - radius).abs() < 2.0);
            }
        }
    }

    #[test]
    fn a_real_notch_is_preserved() {
        let points = vec![
            Point2DMm::new(0.0, 0.0),
            Point2DMm::new(200.0, 0.0),
            Point2DMm::new(200.0, 25.0),
            Point2DMm::new(225.0, 25.0),
            Point2DMm::new(225.0, 0.0),
            Point2DMm::new(400.0, 0.0),
            Point2DMm::new(400.0, 100.0),
            Point2DMm::new(0.0, 100.0),
        ];
        let result = straighten_runs(&points, true).unwrap();
        assert!(result.contains(&Point2DMm::new(200.0, 25.0)));
        assert!(result.contains(&Point2DMm::new(225.0, 25.0)));
    }

    #[test]
    fn sharp_material_edge_replaces_a_soft_shadow_boundary() {
        let mut image = RgbImage::from_pixel(400, 500, image::Rgb([235; 3]));
        for y in 50..450 {
            for x in 80..130 {
                let value = 235 - ((x - 80) * 35 / 50) as u8;
                image.put_pixel(x, y, image::Rgb([value; 3]));
            }
            for x in 130..210 {
                image.put_pixel(x, y, image::Rgb([80; 3]));
            }
        }
        let contour = Contour {
            points: vec![
                Point2DMm::new(100.0, 50.0),
                Point2DMm::new(210.0, 50.0),
                Point2DMm::new(210.0, 450.0),
                Point2DMm::new(100.0, 450.0),
            ],
            closed: true,
        };
        let result = regularize_contours(vec![contour], &image, None).unwrap();
        let points = &result[0].points;
        let crossings: Vec<_> = (0..points.len())
            .filter_map(|i| {
                let a = points[i];
                let b = points[(i + 1) % points.len()];
                if (a.y < 250.0) == (b.y < 250.0) {
                    return None;
                }
                Some(a.x + (250.0 - a.y) * (b.x - a.x) / (b.y - a.y))
            })
            .collect();
        assert_eq!(crossings.len(), 2);
        let left = crossings.iter().copied().fold(f64::INFINITY, f64::min);
        let right = crossings.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!((left - 130.0).abs() < 2.0, "left edge {left}");
        assert!((right - 210.0).abs() < 2.0, "right edge {right}");
    }
}
