use std::{array, fs::{File, OpenOptions}, io::{self, BufRead}, sync::{Arc, RwLock}, thread};

use eframe::*;
use egui::*;
use egui_plot::*;
use regex::Regex;
use screenshots::Screen;
use tokio::{
    io::AsyncBufReadExt
};
use std::io::Write;

const BLOCK_SIZES: [u32; 4] = [128, 256, 512, 1024];
const RESET_ALL: bool = true;

// enum Buffer {
//     CPUStreamCompactionWithScan = 0,
//     CPUStreamCompactionWithoutScan = 1,
//     GPUScanNaive = 2,
//     GPUScanEfficient = 3,
//     GPUStreamCompactionEfficient = 4,
//     GPUScanThrust = 5,
//     CPUScan = 6,
//     BufferCount = 7,
// }

const BUFFER_COUNT: usize = 9;

const BUFFER_TITLES : [&str; BUFFER_COUNT] = [
    "GPU Scan Naive",
    "GPU Scan Work Efficient",
    "GPU Stream Compaction Work Efficient",
    "GPU Scan Thrust",
    "CPU Compaction with Scan",
    "CPU Compaction without Scan",
    "CPU Scan",
    "GPU Stream Compaction Thrust",
    "GPU Scan Thread Efficient"
];


fn main() -> Result<(), eframe::Error> {
    let options = NativeOptions {  
        viewport: ViewportBuilder::default().with_fullscreen(true),
        ..Default::default()
    };

    for i in BLOCK_SIZES {
        if RESET_ALL {
            File::create(format!("profile_output/output_{}.txt", i)).unwrap();
        }
        if let Err(x) = 
            eframe::run_native(&format!("Project 2: Stream Compaction Profiler [Block Size: {i}]"), options.clone(), 
        Box::new(|_| Ok(Box::new(App::new(i))))
            )    
        {
            return Err(x);
        }
    }

    Ok(())
}

pub struct App {
    current: Tab,
    buffers: [Arc<RwLock<Vec<(f32, u32)>>>; BUFFER_COUNT],
    block_size: u32
}

impl App {
    pub fn new(block_size: u32) -> Self {
        let plots = get_plots(block_size);

        Self { current: Tab::Scan, 
            buffers: plots,
            block_size
        }
    }
}

fn get_plots(block_size: u32) -> [Arc<RwLock<Vec<(f32, u32)>>>; BUFFER_COUNT] {
    let buffers: [_; BUFFER_COUNT] = array::from_fn(|_| Arc::new(RwLock::new(Vec::<(f32, u32)>::new())));

    let buffers_clone: [_; BUFFER_COUNT] = array::from_fn(|i| buffers[i].clone());

    thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();

        let file = File::open(format!("profile_output/output_{}.txt", block_size)).expect("Unable to open output file");
        let reader = io::BufReader::new(file);

        let mut prefetch_data: Vec<([f32; BUFFER_COUNT], u32)> = Vec::new();

        for line in reader.lines() {
            let line = line.unwrap();
            let mut elems = line.split_whitespace();

            let mut values = [0.0; BUFFER_COUNT];

            for i in 0..BUFFER_COUNT {
                values[i] = elems.next().unwrap().parse().unwrap();
            }

            let size: u32 = elems.next().unwrap().parse().unwrap();
            prefetch_data.push((values, size));
        }

        for i in 5..=27 {
            let size = 1 << i;

            let prefetch = prefetch_data.iter().find(|x| x.1 == i);

            if let Some(x) = prefetch {
                for i in 0..(BUFFER_COUNT) {
                    buffers_clone[i].write().unwrap().push((x.0[i], x.1));
                }
                continue;
            }

            let buffers_move : [_; BUFFER_COUNT] = array::from_fn(|i| buffers_clone[i].clone());

            rt.block_on(async move {
                let mut file = OpenOptions::new();
                let mut file = file.write(true)   // open for writing
                    .append(true)  // append to the end instead of overwriting
                    .create(true)  // create if it doesn’t exist
                    .open(format!("profile_output/output_{}.txt", block_size)).expect("Unable to open output file");

                let value = run_tests(size, block_size).await;
                for c in 0..(BUFFER_COUNT) {
                    buffers_move[c].write().unwrap().push((value[c], i));
                    write!(file, "{} ", value[c]).unwrap();
                }
                writeln!(file, "{}", i).unwrap();
            });
        }
    });

    buffers
}




#[derive(PartialEq)]
enum Tab {
    Scan,
    StreamCompaction,
}

const COLORS: [Color32; 6] = [Color32::RED, Color32::BLUE, Color32::GREEN, Color32::ORANGE, Color32::WHITE, Color32::YELLOW];


impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("tab_bar").show(ctx, |ui| {
            ui.add_space(2.0); // top padding
            ui.horizontal_wrapped(|ui| {
                ui.selectable_value(&mut self.current, Tab::Scan, "Scan");
                ui.selectable_value(&mut self.current, Tab::StreamCompaction, "StreamCompaction");

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Screenshot").clicked() {
                        let screen = Screen::from_point(0, 0).unwrap();
                        let image = screen.capture().unwrap();
                        let tab = match self.current {
                            Tab::Scan => "scan",
                            Tab::StreamCompaction => "stream_compaction"
                        };
                        let file_name = format!("../img/{}_{}_thread_efficient.png", tab, self.block_size);
                        image.save(file_name).unwrap();
                    }

                    if ui.button("Reset").clicked() {
                        File::create(format!("profile_output/output_{}.txt", self.block_size)).unwrap();
                    }
                });
            });
            ui.add_space(2.0); // top padding
        });

        let buffers: [std::sync::RwLockReadGuard<'_, Vec<(f32, u32)>>; BUFFER_COUNT] = array::from_fn(|i| self.buffers[i].read().unwrap());

        egui::CentralPanel::default().show(ctx, |ui| {
            egui_plot::Plot::new("Plot")
                .allow_drag(false)
                .allow_zoom(false)
                .auto_bounds(Vec2b {
                    x: true, y: true
                })
                .include_x(0.0)
                .include_y(0.0)
                // .include_x(0.0)
                // .include_x(500_00.0)
                // .include_y(0.0)
                // .include_y(600.0)
                .y_axis_label("Algorithm Runtime (ms)")
                .x_axis_label("Array Size (1 << x)")
                .legend(Legend::default().position(Corner::LeftTop))
                .show(ui, |plot_ui|  {
                    let identifier = match self.current {
                        Tab::Scan => |x: &str| x.contains("Scan") && !x.contains("Compaction"),
                        Tab::StreamCompaction => |x: &str| x.contains("Compaction")
                    };

                    buffers.iter()
                        .enumerate()
                        .filter(|(i, _)| identifier(BUFFER_TITLES[*i]))
                        .enumerate()
                        .for_each(|(c, (i, buf))| {
                            plot_ui.line(
                                Line::new(buf.iter().map(|(ms, size)|{
                                    [*size as f64, *ms as f64]
                                }).collect::<Vec<[f64; 2]>>()).color(COLORS[c]).name(BUFFER_TITLES[i])
                            );
                        }); 
                });
        });

        ctx.request_repaint();    
    }
}

async fn run_tests(size: u32, block_size: u32) -> [f32; BUFFER_COUNT] {
    let mut cuda = tokio::process::Command::new("../build/bin/Release/cis5650_stream_compaction_test.exe")
        .current_dir("../")
        .arg("-size")
        .arg(size.to_string())
        .arg("-blocksize")
        .arg(block_size.to_string())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .expect("Failed to start process");

    let stdout = cuda.stdout.take().expect("No stdout captured");
    let mut reader = tokio::io::BufReader::new(stdout).lines();

    let mut values = [0.0_f32; BUFFER_COUNT];

    // Spawn task to print stdout
    while let Ok(Some(line)) = reader.next_line().await {
        if line.starts_with("ERROR") {
            println!("{}", line);
        }

        let re = Regex::new(r"^([^:]+):.*?([0-9.]+)ms").unwrap();
        if let Some(caps) = re.captures(&line) {
            let mut found = false;
            for (i, title) in BUFFER_TITLES.iter().enumerate() {
                if caps[1] == **title {
                    values[i] = caps[2].parse().unwrap();
                    found = true;
                    break;
                }
            }

            if !found {
                println!("{} not found in title", &caps[1]);
            }
        } else {
            println!("{} not captured", line);
        }
    }

    return values;
}
