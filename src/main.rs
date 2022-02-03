extern crate aerso;

// Include constants parsed from YAML file
include!(concat!(env!("OUT_DIR"), "/c_m_delta_elev_fits.rs"));

fn h(x: f64, p: f64) -> f64 {
    h_k(x, p, 10.0)    
}

fn h_k(x: f64, p: f64, k: f64) -> f64 {
    ( 1.0 + (-2.0*k*(x-p)).exp() ).recip()
}

fn s(x: f64, l: f64, h: f64) -> f64 {
    s_k(x, l, h, 10.0)
}

fn s_k(x: f64, l: f64, h: f64, k: f64) -> f64 {
    h_k(x,l,k) * (1.0 - h_k(x,h,k))
}

use aerso::*;
use aerso::types::*;

const S: f64 = 2.625e5 / (1000.0 * 1000.0);
const C: f64 = 0.23;

fn get_dcm_wind2body(airstate: &AirState) -> Matrix3 {
    let ca = airstate.alpha.cos();
    let sa = airstate.alpha.sin();

    let cb = airstate.beta.cos();
    let sb = airstate.beta.sin();

    let body2wind = Matrix3::new(
          ca * cb,   sb,  sa * cb ,
         -ca * sb,   cb, -sa * sb ,
            -sa  ,  0.0,    ca    
        );
    
    body2wind.transpose()
}

struct Lift;
impl<I> AeroEffect<I> for Lift {
    fn get_effect(&self, airstate: AirState, _rates: Vector3, _inputstate: &I) -> (Force,Torque) {
        fn c_l(alpha: f64) -> f64 {
            let cl_0 = 0.16146493;
            let cl_alpha = 5.22123182;
            let pstall = 0.27192924;
            let nstall = -0.38997521;
            s(alpha,nstall,pstall) * (cl_0 + cl_alpha * alpha)
        }
        
        let c_l = c_l(airstate.alpha);
        let lift = airstate.q * S * c_l;
        
        let wind2body = get_dcm_wind2body(&airstate);
        let lift_body = wind2body * Vector3::new(0.0,0.0,-lift);
        
        (Force::body(lift_body[0], lift_body[1], lift_body[2]),Torque::body(0.0,0.0,0.0))
    }
}

struct Drag;
impl<I> AeroEffect<I> for Drag {
    fn get_effect(&self, airstate: AirState, _rates: Vector3, _inputstate: &I) -> (Force,Torque) {
        fn c_d(alpha: f64) -> f64 {
            let cd_0 = 0.06712609;
            let cd_alpha = 2.38136262;
            let alpha_cd0 = 0.02072577;
            let alpha_lim = (30.0f64).to_radians();
            s(alpha,-alpha_lim,alpha_lim) * cd_alpha*(alpha-alpha_cd0).powi(2) + cd_0
            + (1.0 - s(alpha,-alpha_lim,alpha_lim)) * 2.0
        }
        
        let c_d = c_d(airstate.alpha);
        let drag = airstate.q * S * c_d;
        
        let wind2body = get_dcm_wind2body(&airstate);
        let drag_body = wind2body * Vector3::new(-drag,0.0,0.0);
        
        (Force::body(drag_body[0], drag_body[1], drag_body[2]),Torque::body(0.0,0.0,0.0))
    }
}

struct PitchingMoment;
impl AeroEffect<[f64;4]> for PitchingMoment {
    fn get_effect(&self, airstate: AirState, _rates: Vector3, inputstate: &[f64;4]) -> (Force,Torque) {
        fn c_m(alpha: f64) -> f64 {
            let alpha_lim = 15.0f64.to_radians();
            let asymptote = 0.5;
            let k = 12.0;
            
            let cm_0 = 0.0529582 ;
            let alpha_cm0 = 0.04838408;
            let hscale = 1.41514536;
            let vscale = -0.5462109;
            
            s_k(alpha,-alpha_lim,alpha_lim,k) * (vscale * (hscale*(alpha-alpha_cm0)).tan() + cm_0)
                + (1.0-h_k(alpha,-alpha_lim,k)) * asymptote
                + h_k(alpha,alpha_lim,k) * -asymptote
        }
        
        fn c_m_delta_elev_fit(alpha: f64, coeffs: [f64;4]) -> f64 {
            let [a,b,c,d] = coeffs;
            a * (b*alpha + c).tanh() + d
        }
        
        fn c_m_delta_elev(alpha: f64, throttle: f64, airspeed: f64) -> f64 {
            //c_m_delta_elev_fit(alpha,c_m_delta_elev_coeffs::THR_0_2_ASPD_10_0)
            0.0
        }
        
        let c_m = c_m(airstate.alpha) + c_m_delta_elev(airstate.alpha,inputstate[2],airstate.airspeed) * inputstate[1];
        let moment = airstate.q * S * C * c_m;
        
        (Force::body(0.0,0.0,0.0),Torque::body(0.0,moment,0.0))
    }
}

struct PitchDamping;
impl AeroEffect<[f64;4]> for PitchDamping {
    fn get_effect(&self, _airstate: AirState, rates: Vector3, _inputstate: &[f64;4]) -> (Force,Torque) {
        const C_D_FLAT_PLATE: f64 = 1.28;
        
        fn q_at_radius(rate: f64, radius: f64) -> f64 {
            let speed_at_radius = rate * radius;
            0.5 * 1.225 * speed_at_radius.powi(2)
        }
        let hstab_area = 82510.049;
        let hstab_offset = 553.352;
        
        let area = hstab_area / 1000.0f64.powi(2);
        let moment_arm = hstab_offset / 1000.0;
        
        let q = q_at_radius(rates[1], moment_arm);
        let drag = q * area * C_D_FLAT_PLATE;
        
        let moment = -drag * moment_arm - 0.1 * rates[1];
        
        (Force::body(0.0,0.0,0.0),Torque::body(0.0,moment,0.0))
    }
}



fn main() {
    let initial_position = Vector3::zeros();
    let initial_velocity = Vector3::new(15.0,0.0,0.0);
    let initial_attitude = UnitQuaternion::from_euler_angles(0.0,0.0,0.0);
    let initial_rates = Vector3::zeros();
    
    let k_body = Body::new( 1.5, 0.03*Matrix3::identity(), initial_position, initial_velocity, initial_attitude, initial_rates);

    let a_body = AeroBody::new(k_body);
    
    let mut vehicle = AffectedBody {
        body: a_body,
        effectors: vec![Box::new(Lift),Box::new(Drag),Box::new(PitchingMoment),Box::new(PitchDamping)],
        };
    
    
    println!("time,x,y,z,qx,qy,qz,qw,u,v,w,alpha,airspeed,pitching_moment");
            
    let delta_t = 0.01;
    let mut time = 0.0;
    while time < 25.0 {
        vehicle.step(delta_t, &[0.0,0.0,0.0,0.0]);
        time += delta_t;
        let (_,moment) = PitchingMoment{}.get_effect(vehicle.get_airstate(),vehicle.rates(),&[0.0,0.0,0.0,0.0]);
        //println!("{}",vehicle.position());
        //let airstate = vehicle.body.get_airstate();
        //println!("A: {}, B: {}, V: {}, Q: {}",airstate.alpha,airstate.beta,airstate.airspeed,airstate.q);
        println!("{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            time,
            vehicle.body.position()[0],
            vehicle.body.position()[1],
            vehicle.body.position()[2],
            vehicle.body.attitude()[0],
            vehicle.body.attitude()[1],
            vehicle.body.attitude()[2],
            vehicle.body.attitude()[3],
            vehicle.body.velocity()[0],
            vehicle.body.velocity()[1],
            vehicle.body.velocity()[2],
            vehicle.body.get_airstate().alpha,
            vehicle.body.get_airstate().airspeed,
            moment.torque[1],
        );
    }
}
