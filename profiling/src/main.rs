use std::{fs::{File, OpenOptions}, io::{self, BufRead}, sync::{Arc, RwLock}, thread};

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
    cpu_stream_compact: Arc<RwLock<Vec<(f32, u32)>>>,
    gpu_scan_naive: Arc<RwLock<Vec<(f32, u32)>>>,
    gpu_scan_efficient: Arc<RwLock<Vec<(f32, u32)>>>,
    gpu_stream_compact_efficient: Arc<RwLock<Vec<(f32, u32)>>>,
    gpu_scan_thrust: Arc<RwLock<Vec<(f32, u32)>>>,
    block_size: u32
}

impl App {
    pub fn new(block_size: u32) -> Self {
        let plots = get_plots(block_size);

        Self { current: Tab::Scan, 
            cpu_stream_compact: plots.0, 
            gpu_scan_naive: plots.1, 
            gpu_scan_efficient: plots.2,
            gpu_stream_compact_efficient: plots.3,
            gpu_scan_thrust: plots.4,
            block_size
        }
    }
}

fn get_plots(block_size: u32) -> (Arc<RwLock<Vec<(f32, u32)>>>, Arc<RwLock<Vec<(f32, u32)>>>, Arc<RwLock<Vec<(f32, u32)>>>, Arc<RwLock<Vec<(f32, u32)>>>, Arc<RwLock<Vec<(f32, u32)>>>) {
    let cpu_stream_compact = Arc::new(RwLock::new(Vec::<(f32, u32)>::new()));
    let gpu_scan_naive = Arc::new(RwLock::new(Vec::<(f32, u32)>::new()));
    let gpu_scan_efficient = Arc::new(RwLock::new(Vec::<(f32, u32)>::new()));
    let gpu_stream_compact_efficient = Arc::new(RwLock::new(Vec::<(f32, u32)>::new()));
    let gpu_scan_thrust = Arc::new(RwLock::new(Vec::<(f32, u32)>::new()));

    let cpu_stream_compact_clone = cpu_stream_compact.clone();
    let gpu_scan_naive_clone = gpu_scan_naive.clone();
    let gpu_scan_efficient_clone = gpu_scan_efficient.clone();
    let gpu_stream_compact_efficient_clone = gpu_stream_compact_efficient.clone();
    let gpu_scan_thrust_clone = gpu_scan_thrust.clone();


    thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();

        let file = File::open(format!("profile_output/output_{}.txt", block_size)).expect("Unable to open output file");
        let reader = io::BufReader::new(file);

        let mut prefetch_data: Vec<(f32, f32, f32, f32, f32, u32)> = Vec::new();

        for line in reader.lines() {
            let line = line.unwrap();
            let mut elems = line.split_whitespace();
            let cpu_sc_ms: f32 = elems.next().unwrap().parse().unwrap();
            let gpu_scan_ms: f32 = elems.next().unwrap().parse().unwrap();
            let gpu_scan_efficient_ms: f32 = elems.next().unwrap().parse().unwrap();
            let gpu_sc_efficient_ms: f32 = elems.next().unwrap().parse().unwrap();
            let gpu_scan_thrust_ms: f32 = elems.next().unwrap().parse().unwrap();
            let size: u32 = elems.next().unwrap().parse().unwrap();

            prefetch_data.push((cpu_sc_ms, gpu_scan_ms, gpu_scan_efficient_ms, gpu_sc_efficient_ms, gpu_scan_thrust_ms, size));
        }

        for i in 5..=27 {
            let size = 1 << i;

            let prefetch = prefetch_data.iter().find(|x| x.5 == i);

            if let Some(x) = prefetch {
                cpu_stream_compact_clone.write().unwrap().push((x.0, x.5));
                gpu_scan_naive_clone.write().unwrap().push((x.1, x.5));
                gpu_scan_efficient_clone.write().unwrap().push((x.2, x.5));
                gpu_stream_compact_efficient_clone.write().unwrap().push((x.3, x.5));
                gpu_scan_thrust_clone.write().unwrap().push((x.4, x.5));

                continue;
            }

            let cpu_stream_compact_move = cpu_stream_compact_clone.clone();
            let gpu_scan_naive_move = gpu_scan_naive_clone.clone();
            let gpu_scan_efficient_move = gpu_scan_efficient_clone.clone();
            let gpu_stream_compact_efficient_move = gpu_stream_compact_efficient_clone.clone();
            let gpu_scan_thrust_move = gpu_scan_thrust_clone.clone();

            rt.block_on(async move {
                let mut file = OpenOptions::new();
                let mut file = file.write(true)   // open for writing
                    .append(true)  // append to the end instead of overwriting
                    .create(true)  // create if it doesn’t exist
                    .open(format!("profile_output/output_{}.txt", block_size)).expect("Unable to open output file");

                let value = run_tests(size, block_size).await;
                cpu_stream_compact_move.write().unwrap().push((value.0, i));
                gpu_scan_naive_move.write().unwrap().push((value.1, i));
                gpu_scan_efficient_move.write().unwrap().push((value.2, i));
                gpu_stream_compact_efficient_move.write().unwrap().push((value.3, i));
                gpu_scan_thrust_move.write().unwrap().push((value.4, i));
                writeln!(file, "{} {} {} {} {} {}", value.0, value.1, value.2, value.3, value.4, i).unwrap();
            });
        }
    });

    (cpu_stream_compact, gpu_scan_naive, gpu_scan_efficient, gpu_stream_compact_efficient, gpu_scan_thrust)
}




#[derive(PartialEq)]
enum Tab {
    Scan,
    StreamCompaction,
}

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
                        let file_name = format!("../img/{}_{}.png", tab, self.block_size);
                        image.save(file_name).unwrap();
                    }

                    if ui.button("Reset").clicked() {
                        File::create(format!("profile_output/output_{}.txt", self.block_size)).unwrap();
                    }
                });
            });
            ui.add_space(2.0); // top padding
        });

        
        let cpu_stream_compact = self.cpu_stream_compact.read().unwrap();
        let gpu_scan_naive = self.gpu_scan_naive.read().unwrap();
        let gpu_scan_efficient = self.gpu_scan_efficient.read().unwrap();
        let gpu_stream_compact_efficient = self.gpu_stream_compact_efficient.read().unwrap();
        let gpu_scan_thrust = self.gpu_scan_thrust.read().unwrap();
        
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
                    match self.current {
                        Tab::Scan => {
                            plot_ui.line(
                                Line::new(gpu_scan_naive.iter().map(|(ms, size)|{
                                    [*size as f64, *ms as f64]
                                }).collect::<Vec<[f64; 2]>>()).color(Color32::RED).name("GPU Scan Naive")
                            );

                            plot_ui.line(
                                Line::new(gpu_scan_efficient.iter().map(|(ms, size)|{
                                    [*size as f64, *ms as f64]
                                }).collect::<Vec<[f64; 2]>>()).color(Color32::BLUE).name("GPU Scan Efficient")
                            );

                            plot_ui.line(
                                Line::new(gpu_scan_thrust.iter().map(|(ms, size)|{
                                    [*size as f64, *ms as f64]
                                }).collect::<Vec<[f64; 2]>>()).color(Color32::GREEN).name("GPU Scan Thrust")
                            );
                        },
                        Tab::StreamCompaction => {
                            plot_ui.line(
                                Line::new(gpu_stream_compact_efficient.iter().map(|(ms, size)|{
                                    [*size as f64, *ms as f64]
                                }).collect::<Vec<[f64; 2]>>()).color(Color32::RED).name("GPU Stream Compaction Efficient")
                            );

                            plot_ui.line(
                                Line::new(cpu_stream_compact.iter().map(|(ms, size)|{
                                    [*size as f64, *ms as f64]
                                }).collect::<Vec<[f64; 2]>>()).color(Color32::BLUE).name("CPU Stream Compaction")
                            );
                        }
                    }

                    
                });
        });

        ctx.request_repaint();    
    }
}

async fn run_tests(size: u32, block_size: u32) -> (f32, f32, f32, f32, f32) {
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

    let mut cpu_stream_compact: f32 = -1.0;
    let mut gpu_scan_naive: f32 = -1.0;
    let mut gpu_scan_efficient: f32 = -1.0;
    let mut gpu_stream_compact_efficient: f32 = -1.0;
    let mut gpu_scan_thrust: f32 = -1.0;

    // Spawn task to print stdout
    while let Ok(Some(line)) = reader.next_line().await {
        if line.starts_with("ERROR") {
            println!("{}", line);
        }

        let re = Regex::new(r"^([^:]+):.*?([0-9.]+)ms").unwrap();
        if let Some(caps) = re.captures(&line) {
            match &caps[1] {
                "CPU Compaction with Scan" => cpu_stream_compact = caps[2].parse().unwrap(),
                "CPU Compaction without Scan" => (),
                "CPU Scan" => (),
                "CPU Compaction" => (),
                "GPU Scan Naive" => gpu_scan_naive = caps[2].parse().unwrap(),
                "GPU Scan Work Efficient" => gpu_scan_efficient = caps[2].parse().unwrap(),
                "GPU Compaction Work Efficient" => gpu_stream_compact_efficient = caps[2].parse().unwrap(),
                "GPU Thrust Scan" => gpu_scan_thrust = caps[2].parse().unwrap(),
                _ => println!("Unaccounted for string: {}", &caps[1])
            }
        } else {
            println!("{} not captured", line);
        }
    }

    return (cpu_stream_compact, gpu_scan_naive, gpu_scan_efficient, gpu_stream_compact_efficient, gpu_scan_thrust);
}
