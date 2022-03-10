extern crate aerso;

use aerso::types::*;
use aerso::AeroEffect;

extern crate mxsrs;

use mxsrs::{MXS,PitchingMoment,Lift};

fn main() {
    let initial_position = Vector3::zeros();
    let initial_velocity = Vector3::new(11.0,0.0,0.0);
    let initial_attitude = UnitQuaternion::from_euler_angles(0.0,6.0f64.to_radians(),0.0);
    let initial_rates = Vector3::zeros();
    
    let mut vehicle = MXS::new_with_state(initial_position, initial_velocity, initial_attitude, initial_rates);
    
    
    println!("time,x,y,z,qx,qy,qz,qw,u,v,w,alpha,airspeed,pitching_moment,lift");
            
    let delta_t = 0.01;
    let mut time = 0.0;
    while time < 50.0 {
        let elevator = 0.0; //if (12.0..20.0).contains(&time) { 2.0f64.to_radians() } else { 0.0 };
        let throttle = 0.23;
        let input = [0.0,elevator,throttle,0.0];
        vehicle.0.step(delta_t, &input);
        time += delta_t;
        let (_,moment) = PitchingMoment{}.get_effect(vehicle.0.get_airstate(),vehicle.0.rates(),&input);
        let (lift,_) = Lift{}.get_effect(vehicle.0.get_airstate(),vehicle.0.rates(),&input);
        //println!("{}",vehicle.position());
        //let airstate = vehicle.body.get_airstate();
        //println!("A: {}, B: {}, V: {}, Q: {}",airstate.alpha,airstate.beta,airstate.airspeed,airstate.q);
        println!("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            time,
            vehicle.0.body.position()[0],
            vehicle.0.body.position()[1],
            vehicle.0.body.position()[2],
            vehicle.0.body.attitude()[0],
            vehicle.0.body.attitude()[1],
            vehicle.0.body.attitude()[2],
            vehicle.0.body.attitude()[3],
            vehicle.0.body.velocity()[0],
            vehicle.0.body.velocity()[1],
            vehicle.0.body.velocity()[2],
            vehicle.0.body.get_airstate().alpha,
            vehicle.0.body.get_airstate().airspeed,
            moment.torque[1],
            lift.force.norm(),
        );
    }
}
