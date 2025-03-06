use opencv::core::{
    self,
    Mat,
    MatExpr,
    MatExprTraitConst,
    MatTraitConst,
    Point,
    Vec4i,
    Vector,
    Scalar,
    Ptr,
};
use opencv::imgproc::{self, adaptive_threshold, erode, morphology_ex, dilate, find_contours, draw_contours, bounding_rect};
use opencv::Error;
use opencv::hub_prelude::BackgroundSubtractorMOG2Trait;
use crate::util::video_frames::{FrameProcessor, VideoFrames};

const SAMPLE_COUNT: usize = 10;

pub struct MotionMog2 {

    bg_remover: Ptr<opencv::video::BackgroundSubtractorMOG2>,

    diff_mask: Mat,                 // MOG2 motion mask (source for frame copy operations)
    dst_frame: Mat,                 // destination for frame copy operations

    default_scalar: Scalar,

    mog2_learning_rate: f64,

    adaptive_max_value: f64,
    adaptive_method: i32,
    adaptive_type: i32,
    adaptive_block_size: i32,
    adaptive_c: f64,

    erode_iterations: i32,
    erode_border_type: i32,
    erode_kernel: MatExpr,
    erode_anchor_point: Point,

    dilate_iterations: i32,
    dilate_border_type: i32,
    dilate_kernel: MatExpr,
    dilate_anchor_point: Point,

    close_operation: i32,
    close_iterations: i32,
    close_border_type: i32,
    close_kernel: MatExpr,
    close_anchor_point: Point,

    contour_mode: i32,
    contour_method: i32,
    contours: Vector<Vector<Point>>,
    contour_anchor_point: Point,
    contour_fill_anchor_point: Point,
    contour_index: i32,
    contour_thickness: i32,
    contour_line_type: i32,
    contour_max_level: i32,
    contour_color: Scalar,

    bounding_boxes: Vec<core::Rect>,
    bounding_box_min_area: i32,

    mog2_values: Vec<f64>,

    _total_area: i32,
    _contour_area: i32,
    _contour_p1: Point,
    _contour_p2: Point,
}

impl<'a> MotionMog2 {
    pub fn new(
        prv_frame: &'a Mat,
        history: i32,
        var_threshold: f64,
        adaptive_block_size: i32,
        adaptive_c: f64,
    ) -> Self {
        Self {
            diff_mask: Mat::zeros(
                prv_frame.rows(),
                prv_frame.cols(),
                prv_frame.typ(),
            ).unwrap().to_mat().unwrap(),
            dst_frame: Mat::zeros(
                prv_frame.rows(),
                prv_frame.cols(),
                prv_frame.typ(),
            ).unwrap().to_mat().unwrap(),

            default_scalar: Scalar::default(),

            mog2_learning_rate: -1.0,
            bg_remover: opencv::video::create_background_subtractor_mog2(
                history,
                var_threshold,
                false
            ).unwrap(),

            adaptive_max_value: 255.0,
            adaptive_method: imgproc::ADAPTIVE_THRESH_GAUSSIAN_C,
            adaptive_type: imgproc::THRESH_BINARY_INV,
            adaptive_block_size,
            adaptive_c,

            erode_kernel: Mat::ones(
                2,
                2,
                core::CV_8U
            ).unwrap(),
            erode_anchor_point: Point::new(-1, -1),
            erode_iterations: 1,
            erode_border_type: core::BORDER_CONSTANT,

            dilate_kernel: Mat::ones(
                2,
                2,
                core::CV_8U,
            ).unwrap(),
            dilate_anchor_point: Point::new(-1, -1),
            dilate_iterations: 1,
            dilate_border_type: core::BORDER_CONSTANT,

            close_operation: imgproc::MORPH_CLOSE,
            close_kernel: Mat::ones(
                3,
                3,
                core::CV_8U,
            ).unwrap(),
            close_anchor_point: Point::new(-1, -1),
            close_iterations: 1,
            close_border_type: core::BORDER_CONSTANT,

            contour_mode: imgproc::RETR_EXTERNAL,
            contour_method: imgproc::CHAIN_APPROX_SIMPLE,
            contours: Vector::new(),
            contour_color: Scalar::new(
                255.0,
                255.0,
                255.0,
                0.0
            ),
            contour_anchor_point: Point::new(-1, -1),
            contour_index: -1,
            contour_thickness: imgproc::FILLED,
            contour_line_type: imgproc::LINE_8,
            contour_max_level: 0,
            contour_fill_anchor_point: Point::new(-1, -1),

            bounding_boxes: Vec::with_capacity(10),
            bounding_box_min_area: 500,

            mog2_values: vec![0.0; SAMPLE_COUNT],

            _total_area: 0,
            _contour_area: 0,
            _contour_p1: Point::new(-1, -1),
            _contour_p2: Point::new(-1, -1),
        }
    }

    pub fn default() -> Self {
        Self {

            diff_mask: Mat::default(),
            dst_frame: Mat::default(),

            default_scalar: Scalar::default(),

            mog2_learning_rate: -1.0,
            bg_remover: opencv::video::create_background_subtractor_mog2(
                0,
                0.0,
                false
            ).unwrap(),

            adaptive_max_value: 255.0,
            adaptive_method: imgproc::ADAPTIVE_THRESH_GAUSSIAN_C,
            adaptive_type: imgproc::THRESH_BINARY_INV,
            adaptive_block_size: 17,
            adaptive_c: 9.0,

            erode_kernel: Mat::ones(
                2,
                2,
                core::CV_8U
            ).unwrap(),
            erode_anchor_point: Point::new(-1, -1),
            erode_iterations: 1,
            erode_border_type: core::BORDER_CONSTANT,

            dilate_kernel: Mat::ones(
                2,
                2,
                core::CV_8U,
            ).unwrap(),
            dilate_anchor_point: Point::new(-1, -1),
            dilate_iterations: 1,
            dilate_border_type: core::BORDER_CONSTANT,

            close_operation: imgproc::MORPH_CLOSE,
            close_kernel: Mat::ones(
                3,
                3,
                core::CV_8U,
            ).unwrap(),
            close_anchor_point: Point::new(-1, -1),
            close_iterations: 1,
            close_border_type: core::BORDER_CONSTANT,

            contour_mode: imgproc::RETR_EXTERNAL,
            contour_method: imgproc::CHAIN_APPROX_SIMPLE,
            contours: Vector::new(),
            contour_color: Scalar::new(
                255.0,
                255.0,
                255.0,
                0.0
            ),
            contour_anchor_point: Point::new(-1, -1),
            contour_index: -1,
            contour_thickness: imgproc::FILLED,
            contour_line_type: imgproc::LINE_8,
            contour_max_level: 0,
            contour_fill_anchor_point: Point::new(-1, -1),

            bounding_boxes: Vec::with_capacity(10),
            bounding_box_min_area: 500,

            mog2_values: vec![0.0; SAMPLE_COUNT],

            _total_area: 0,
            _contour_area: 0,
            _contour_p1: Point::new(-1, -1),
            _contour_p2: Point::new(-1, -1),
        }
    }

    pub fn get_diff_mask(&self) -> &Mat {
        &self.diff_mask
    }

    pub fn get_contours(&self) -> &Vector<Vector<Point>> {
        &self.contours
    }

    pub fn get_values(&self) -> &Vec<f64> {
        &self.mog2_values
    }

    pub fn get_bounding_boxes(&self) -> &Vec<core::Rect> {
        &self.bounding_boxes
    }

    fn process_frame(&mut self, cur_frame: &'a Mat) -> Result<(), Error> {

        self.contours.clear();

        self.bg_remover.apply(
            cur_frame,
            &mut self.diff_mask,
            self.mog2_learning_rate
        )?;

        adaptive_threshold(
            &self.diff_mask,
            &mut self.dst_frame,
            self.adaptive_max_value,
            self.adaptive_method,
            self.adaptive_type,
            self.adaptive_block_size,
            self.adaptive_c,
        )?;

        erode(
            &self.dst_frame,
            &mut self.diff_mask,
            &self.erode_kernel,
            self.erode_anchor_point,
            self.erode_iterations,
            self.erode_border_type,
            self.default_scalar,
        )?;

        dilate(
            &self.dst_frame,
            &mut self.diff_mask,
            &self.dilate_kernel,
            self.dilate_anchor_point,
            self.dilate_iterations,
            self.dilate_border_type,
            self.default_scalar,
        )?;

        morphology_ex(
            &self.diff_mask,
            &mut self.dst_frame,
            self.close_operation,
            &self.close_kernel,
            self.close_anchor_point,
            self.close_iterations,
            self.close_border_type,
            self.default_scalar,
        )?;

        find_contours(
            &self.dst_frame,
            &mut self.contours,
            self.contour_mode,
            self.contour_method,
            self.contour_anchor_point,
        )?;

        // filter bounding boxes
        self.bounding_boxes = self.contours.iter()
            .filter_map(|c| bounding_rect(&c).ok())
            .filter(|rect| rect.area() >= self.bounding_box_min_area)
            .collect();

        // Sort rectangles by area (largest first) to optimize nesting check
        self.bounding_boxes.sort_unstable_by_key(|rect| -(rect.area() as i64));

        draw_contours(
            &mut self.diff_mask,
            &self.contours,
            self.contour_index,
            self.contour_color,
            self.contour_thickness,
            self.contour_line_type,
            &Vector::<Vec4i>::new(),
            self.contour_max_level,
            self.contour_fill_anchor_point,
        )?;

        Ok(())
    }

    fn _calc_total_area(&mut self) -> Result<(), Error> {
        self._total_area = 0;
        self._contour_area = 0;
        for index in 0..self.contours.len() {
            if let Ok(contour) = self.contours.get(index) {
                for i in 0..contour.len() {
                    self._contour_p1 = contour.get(i)?;
                    self._contour_p2 = contour.get((i + 1) % contour.len())?;
                    self._contour_area += (self._contour_p1.x - self._contour_p2.x) * (self._contour_p1.y + self._contour_p2.y);
                }
                self. _total_area += self._contour_area.abs();
            }
        }
        self.mog2_values.remove(0);
        self.mog2_values.push(self._total_area as f64);
        Ok(())
    }

    pub fn get_area(&self) -> f64 {
        self.mog2_values[SAMPLE_COUNT - 1]
    }

    pub fn get_area_avg(&mut self) -> f64 {
        self._calc_total_area().unwrap();
        self.mog2_values.iter().sum::<f64>() / SAMPLE_COUNT as f64
    }
}

impl<'a> FrameProcessor<'a> for MotionMog2 {
    fn update(&mut self, video_frames: &VideoFrames) -> Result<(), Error> {
        self.process_frame(&video_frames.mono.quarter.cur)
    }
}