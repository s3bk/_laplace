#![feature(test, fixed_size_array, core_intrinsics)]
extern crate test;
use test::Bencher;
use std::ops::{Deref, Index};
use std::array::FixedSizeArray;

mod fast;
use fast::fff;

#[inline(always)]
pub fn square<T>(i: T) -> T 
where 
    T:std::ops::Mul + std::ops::Mul<Output = T> + Copy {
    i*i
}

#[derive(Debug)]
pub struct Laplace2dMatrix {
    pub n_x       : usize ,
    pub n_y       : usize ,
    pub n         : usize ,
    pub diag      : fff   ,
    pub tri_diag  : fff   ,
    pub side_diag : fff   ,
}

impl Laplace2dMatrix {
    pub fn rectangular(n_x: usize, n_y: usize) -> Laplace2dMatrix {
        Laplace2dMatrix {
                  n_x: n_x                                  ,
                  n_y: n_y                                  ,
                    n: n_x*n_y                              ,
                 diag: fff(-2.0) * fff((square(n_x+1) + square(n_y+1)) as f64),
             tri_diag: fff(square(n_x+1) as f64),
            side_diag: fff(square(n_y+1) as f64),
        }
    }

    pub fn quadratic(n_xy: usize) -> Laplace2dMatrix {
        Laplace2dMatrix::rectangular(n_xy, n_xy)
    }
}

struct Arr<'a> (&'a [fff]);
impl<'a> Index<usize> for Arr<'a> {
    type Output = fff;
    fn index(&self, idx: usize) -> &Self::Output {
        unsafe {
            self.0.get_unchecked(idx)
        }
    }
}

/// calculates the residual r^2 = ||l*x - b||_2^2 
#[inline(never)]
pub fn calculate_residual_squared_quadratic(n_xy: usize, x: &[fff], b: &[fff]) -> fff {
    let l = Laplace2dMatrix::quadratic(n_xy);
    _calculate_residual_squared(&l, x, b)
}

/// calculates the residual r^2 = ||l*x - b||_2^2 
#[inline(never)]
pub fn calculate_residual_squared(l: &Laplace2dMatrix, x: &[fff], b: &[fff]) -> fff {
    _calculate_residual_squared(l, x, b)
}

#[inline(always)]
fn _calculate_residual_squared(l: &Laplace2dMatrix, x: &[fff], b: &[fff]) -> fff {
    // Assumptions that must hold for l
    // Laplace2dMatrix represents a block diagonal matrix where the block matrix
    // is the same for all blocks and the block matrix is a band matrix
    // n_x and n_y are the size of the block matrix and l is a n x n matrix 
    // where n = n_x*n_y 
    assert!(l.n == x.len());
    assert!(l.n == b.len());
    assert!(l.n == l.n_x*l.n_y);
    assert!(l.n > 0);

    let x = Arr(x);
    let b = Arr(b);

    let mut r2 = fff(0.0);

    r2 += square(-b[1-1]  + x[1-1]*l.diag + x[2-1]   * l.tri_diag + x[1+l.n_x-1] * l.side_diag);
    for i in 2 .. l.n_x {
        r2 += square(-b[i-1] + x[i-1]*l.diag + x[i+1-1]*l.tri_diag + x[i-1-1]*l.tri_diag + x[i+l.n_x-1]*l.side_diag);
    }
    r2 += square(-b[l.n_x-1] + x[l.n_x-1]*l.diag + x[l.n_x-1-1]*l.tri_diag + x[l.n_x+l.n_x-1]*l.side_diag);

    for outer_base in (l.n_x .. l.n_x*(l.n_y-2) + 1).step_by(l.n_x) {
        r2 += square(-b[outer_base+1 -1] + x[outer_base+1 -1]*l.diag + x[outer_base+2   -1]*l.tri_diag + x[outer_base+1-l.n_x -1]*l.side_diag + x[outer_base+1+l.n_x-1]*l.side_diag);
        for i in 2 .. l.n_x {
            r2 += square(-b[outer_base+i-1] + x[outer_base+i-1]*l.diag + x[outer_base+i+1-1]*l.tri_diag + x[outer_base+i-1-1]*l.tri_diag + x[outer_base+i+l.n_x-1]*l.side_diag + x[outer_base+i-l.n_x-1]*l.side_diag);
        }
        r2 += square(-b[outer_base+l.n_x-1] + x[outer_base+l.n_x-1]*l.diag + x[outer_base+l.n_x-1-1]*l.tri_diag + x[outer_base+l.n_x+l.n_x-1]*l.side_diag + x[outer_base     -1]*l.side_diag);
    }

    r2 += square(-b[l.n-l.n_x+1-1] + x[l.n-l.n_x+1-1]*l.diag + x[l.n-l.n_x+2-1]*l.tri_diag + x[l.n-l.n_x-l.n_x+1-1]*l.side_diag);
    for i in l.n-l.n_x+2 .. l.n {
        r2 += square(-b[i-1] + x[i-1]*l.diag + x[i+1-1]*l.tri_diag + x[i-1-1]*l.tri_diag + x[i-l.n_x-1]*l.side_diag);
    }
    r2 += square(-b[l.n     -1] + x[l.n     -1]*l.diag + x[l.n-1   -1]*l.tri_diag + x[l.n-l.n_x     -1]*l.side_diag);


    r2 / fff(l.n as f64)
}

#[bench]
fn b1(bencher: &mut Bencher) {
    let l = Laplace2dMatrix::quadratic(10);
    let x = vec![fff(1.2); 100];
    let b = vec![fff(1.2); 100];

    bencher.iter(|| calculate_residual_squared(&l, &x, &b));
}

#[bench]
fn b2(bencher: &mut Bencher) {
    let x = vec![fff(1.2); 100];
    let b = vec![fff(1.2); 100];

    bencher.iter(|| calculate_residual_squared_quadratic(10, &x, &b));
}

fn main() {
    let l = Laplace2dMatrix::quadratic(10);
    let x = vec![fff(1.2); 100];
    let b = vec![fff(1.2); 100];
    let mut sum = fff(0.0);
    for _ in 0 .. 100000000 {
        sum += calculate_residual_squared(&l, &x, &b);
    }
    println!("squared residual: {:?}", sum / 100.);
}
