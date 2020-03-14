use std::ops::*;
use std::fmt;
use std::fmt::Write;
use std::cmp::Ordering;

#[allow(non_camel_case_types)]
#[derive(Copy, Clone, PartialEq, PartialOrd)]
#[repr(C)]
/// Fast, finite, floating-point
pub struct fff(pub f64);

/// Not all float functions are wrapped/implemented in a wrapped/fast version,
/// but all should work by falling back to a regular f64 via Deref.
impl fff {
    #[inline(always)]
    pub fn powf<V: Into<f64>>(self, v: V) -> Self {
        fff(self.0.powf(v.into()))
    }

    #[inline(always)]
    pub fn powi(self, v: i32) -> Self {
        fff(self.0.powi(v))
    }

    #[inline(always)]
    pub fn sqrt(self) -> Self {
        fff(unsafe{std::intrinsics::sqrtf64(self.0)})
    }

    #[inline(always)]
    /// Very slow. Use trunc()
    pub fn round(self) -> Self {
        self.0.round().into()
    }

    #[inline(always)]
    /// Very slow. Use trunc()
    pub fn floor(self) -> Self {
        self.0.floor().into()
    }

    #[inline(always)]
    /// Very slow. Use trunc()
    pub fn ceil(self) -> Self {
        self.0.ceil().into()
    }

    #[inline(always)]
    /// Inaccurate for values that don't fit in i32
    pub fn trunc(self) -> Self {
        fff(self.0 as i32 as f64)
    }

    #[inline(always)]
    pub fn abs(self) -> Self {
        self.0.abs().into()
    }
}

macro_rules! impl_fast {
    ($tr:ident, $fn:ident, $func:ident) => {
        impl $tr for fff {
            type Output = fff;

            #[inline(always)]
            fn $fn(self, other: fff) -> Self::Output {
                unsafe {
                    fff(std::intrinsics::$func(self.0, other.0))
                }
            }
        }

        impl $tr<f64> for fff {
            type Output = fff;

            #[inline(always)]
            fn $fn(self, other: f64) -> Self::Output {
                unsafe {
                    std::intrinsics::$func(self.0, other).into()
                }
            }
        }

        impl $tr<fff> for f64 {
            type Output = fff;

            #[inline(always)]
            fn $fn(self, other: fff) -> Self::Output {
                unsafe {
                    std::intrinsics::$func(self, other.0).into()
                }
            }
        }
    }
}

macro_rules! impl_assign {
    ($tr:ident, $func:ident, $fn:ident) => {
        impl $tr for fff {
            #[inline(always)]
            fn $fn(&mut self, other: fff) {
                *self = self.$func(other)
            }
        }
    }
}

impl_fast! {Add, add, fadd_fast}
impl_assign! {AddAssign, add, add_assign}
impl_fast! {Sub, sub, fsub_fast}
impl_assign! {SubAssign, sub, sub_assign}
impl_fast! {Mul, mul, fmul_fast}
impl_fast! {Rem, rem, frem_fast}
impl_fast! {Div, div, fdiv_fast}

impl Neg for fff {
    type Output = fff;
    fn neg(self) -> Self::Output {
        fff(self.0.neg())
    }
}

impl Eq for fff {
}

impl PartialEq<f64> for fff {
    #[inline(always)]
    fn eq(&self, other: &f64) -> bool {
        self.0.eq(other)
    }
    #[inline(always)]
    fn ne(&self, other: &f64) -> bool {
        self.0.ne(other)
    }
}

impl PartialEq<fff> for f64 {
    #[inline(always)]
    fn eq(&self, other: &fff) -> bool {
        self.eq(&other.0)
    }
    #[inline(always)]
    fn ne(&self, other: &fff) -> bool {
        self.ne(&other.0)
    }
}

impl Ord for fff {
    #[inline(always)]
    fn cmp(&self, other: &fff) -> Ordering {
        self.0.partial_cmp(&other.0).expect("fff")
    }
}

impl PartialOrd<f64> for fff {
    #[inline(always)]
    fn partial_cmp(&self, other: &f64) -> Option<Ordering> {
        self.0.partial_cmp(other)
    }
}

impl PartialOrd<fff> for f64 {
    #[inline(always)]
    fn partial_cmp(&self, other: &fff) -> Option<Ordering> {
        self.partial_cmp(&other.0)
    }
}

impl From<f64> for fff {
    #[inline(always)]
    fn from(v: f64) -> Self {
        debug_assert!(v.is_finite());
        fff(v)
    }
}

impl From<fff> for f64 {
    #[inline(always)]
    fn from(v: fff) -> Self {
        v.0
    }
}

impl Deref for fff {
    type Target = f64;

    #[inline(always)]
    fn deref(&self) -> &f64 {
        &self.0
    }
}

impl fmt::Display for fff {
    #[inline(always)]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(fmt)
    }
}

impl fmt::Debug for fff {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(fmt)?;
        fmt.write_char('f')
    }
}
