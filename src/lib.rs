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

pub struct Lift;
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
        
        (Force::body_vec(lift_body),Torque::body(0.0,0.0,0.0))
    }
}

pub struct Drag;
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
        
        (Force::body_vec(drag_body),Torque::body(0.0,0.0,0.0))
    }
}

pub struct PitchingMoment;
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
        
        fn c_m_delta_elev_fit(elev: f64, coeffs: [f64;4]) -> f64 {
            let [a,b,c,d] = coeffs;
            a * (b*elev + c).tanh() + d
        }
        
        fn c_m_delta_elev(elev: f64, throttle: f64, _airspeed: f64) -> f64 {
            fn interp(x: f64, xl: f64, xh: f64, vl: f64, vh: f64) -> f64 {
                let fraction = (x-xl)/(xh-xl);
                vl + (vh-vl) * fraction
            }
            if throttle <= 0.2 {
                c_m_delta_elev_fit(elev,c_m_delta_elev_coeffs::THR_0_2_ASPD_15_0)
            }
            else if throttle <= 0.35 {
                interp(
                    throttle, 0.2, 0.35,
                    c_m_delta_elev_fit(elev,c_m_delta_elev_coeffs::THR_0_2_ASPD_15_0),
                    c_m_delta_elev_fit(elev,c_m_delta_elev_coeffs::THR_0_35_ASPD_15_0)
                )
            }
            else if throttle <= 0.5 {
                interp(
                    throttle, 0.35, 0.5,
                    c_m_delta_elev_fit(elev,c_m_delta_elev_coeffs::THR_0_35_ASPD_15_0),
                    c_m_delta_elev_fit(elev,c_m_delta_elev_coeffs::THR_0_5_ASPD_15_0)
                )
            }
            else if throttle <= 0.65 {
                interp(
                    throttle, 0.5, 0.65,
                    c_m_delta_elev_fit(elev,c_m_delta_elev_coeffs::THR_0_5_ASPD_15_0),
                    c_m_delta_elev_fit(elev,c_m_delta_elev_coeffs::THR_0_65_ASPD_15_0)
                )
            }
            else if throttle <= 0.8 {
                interp(
                    throttle, 0.65, 0.8,
                    c_m_delta_elev_fit(elev,c_m_delta_elev_coeffs::THR_0_65_ASPD_15_0),
                    c_m_delta_elev_fit(elev,c_m_delta_elev_coeffs::THR_0_8_ASPD_15_0)
                )
            }
            else {
                c_m_delta_elev_fit(elev,c_m_delta_elev_coeffs::THR_0_8_ASPD_15_0)
            }
        }
        
        let c_m = c_m(airstate.alpha) + c_m_delta_elev(inputstate[1],inputstate[2],airstate.airspeed);
        let moment = airstate.q * S * C * c_m;
        
        (Force::body(0.0,0.0,0.0),Torque::body(0.0,moment,0.0))
    }
}

pub struct PitchDamping;
impl AeroEffect<[f64;4]> for PitchDamping {
    fn get_effect(&self, airstate: AirState, rates: Vector3, _inputstate: &[f64;4]) -> (Force,Torque) {
        const HSTAB_OFFSET: f64 = -553.352 / 1000.0;
        const CG_OFFSET: f64 = -0.02;
        const HSTAB_AREA: f64 = 82510.049 / (1000.0*1000.0);
        
        fn cl_t(alpha: f64) -> f64 {
            3.5810 * s(alpha,-0.1745,0.1745) * alpha
             + 0.65 * h(alpha,0.1745)
             - 0.65 * (1.0-h(alpha,-0.1745))
        }
        
        fn alpha_t(pitch_rate: f64, alpha: f64, v_inf: f64) -> f64 {
            (
                (-pitch_rate*HSTAB_OFFSET/1000.0 + v_inf*alpha.sin())
                /(v_inf*alpha.cos())
            ).atan()
        }
        
        let alpha_t = alpha_t(rates[1], airstate.alpha, airstate.airspeed);
        
        let moment = (HSTAB_OFFSET - CG_OFFSET) * airstate.q * HSTAB_AREA * (cl_t(alpha_t) - cl_t(airstate.alpha));
        
        (Force::body(0.0,0.0,0.0),Torque::body(0.0,moment,0.0))
    }
}

pub struct Thrust;
impl AeroEffect<[f64;4]> for Thrust {
    fn get_effect(&self, _airstate: AirState, _rates: Vector3, inputstate: &[f64;4]) -> (Force,Torque) {
        let throttle = inputstate[2];
        let pwm = (throttle * 1000.0) + 1000.0;
        let thrust = -4.120765323840711e-05 * pwm.powi(2) + 0.14130986760422384 * pwm - 110.0;
        
        (Force::body(thrust,0.0,0.0),Torque::body(0.0,0.0,0.0))
    }
}


pub struct MXS(pub AffectedBody<[f64;4]>);

impl MXS {
    pub fn new() -> MXS {
        let initial_position = Vector3::zeros();
        let initial_velocity = Vector3::zeros();
        let initial_attitude = UnitQuaternion::from_euler_angles(0.0,6.0f64.to_radians(),0.0);
        let initial_rates = Vector3::zeros();
        
        Self::new_with_state(initial_position, initial_velocity, initial_attitude, initial_rates)
    }
    
    pub fn new_with_state(initial_position: Vector3, initial_velocity: Vector3, initial_attitude: UnitQuaternion, initial_rates: Vector3) -> MXS {
        
        let k_body = Body::new( 1.5, 0.05*Matrix3::identity(), initial_position, initial_velocity, initial_attitude, initial_rates);

        let a_body = AeroBody::new(k_body);
        
        MXS {
            0: AffectedBody {
                body: a_body,
                effectors: vec![Box::new(Lift),Box::new(Drag),Box::new(PitchingMoment),Box::new(PitchDamping),Box::new(Thrust)],
            }
        }
    }
}
