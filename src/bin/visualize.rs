use wgpu_n_body::{
    inits, runners,
    sims::NaiveSim,
    sims::{self, TreeSim},
};

use winit::{
    dpi::LogicalSize,
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_inner_size(LogicalSize::new(400, 400))
        .build(&event_loop)
        .unwrap();
    let mut should_render = true;
    window.focus_window();

    let sim_params = sims::SimParams {
        particle_num: 5000,
        g: 0.00001,
        e: 0.0001,
        dt: 0.016,
    };
    let mut state = pollster::block_on(runners::OnlineRenderer::<TreeSim>::new(
        &window,
        sim_params,
        inits::disc_init,
    ))
    .unwrap();

    event_loop.run(move |event, _, control_flow| match event {
        Event::RedrawRequested(window_id) if window_id == window.id() => {
            state.update();
            match state.render() {
                Ok(_) => {}
                // Reconfigure the surface if lost
                Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                // The system is out of memory, we should probably quit
                Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                // All other errors (Outdated, Timeout) should be resolved by the next frame
                Err(e) => eprintln!("{:?}", e),
            }
            //std::process::exit(0);
        }
        Event::MainEventsCleared => {
            // RedrawRequested will only trigger once, unless we manually
            // request it.
            if should_render {
                window.request_redraw();
            }
        }
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == window.id() => match event {
            WindowEvent::Focused(focus) => {
                should_render = *focus;
                *control_flow = match should_render {
                    true => ControlFlow::Poll,
                    false => ControlFlow::Wait,
                };
            }
            WindowEvent::Resized(new_size) => {
                state.resize(*new_size);
            }
            WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                state.resize(**new_inner_size);
            }
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        state: ElementState::Pressed,
                        virtual_keycode: Some(VirtualKeyCode::Escape),
                        ..
                    },
                ..
            } => *control_flow = ControlFlow::Exit,
            _ => {
                state.input(event);
            }
        },
        _ => {}
    });
}
