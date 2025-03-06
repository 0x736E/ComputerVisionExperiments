mod detectors;
mod util;
mod masks;

use std::path::Path;
use structopt::StructOpt;
use crate::util::video_processor::{ VideoProcessor, VideoConfig };

fn main() {

    let mut video_proc = VideoProcessor::new();
    let conf = VideoConfig::from_args();
    let path = Path::new(conf.input.to_str().unwrap());
    video_proc.load_videos(path, &conf).unwrap();

}
