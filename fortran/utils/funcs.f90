!////////////////////////////////////////////////////!
! * Intrinsic Special Functions
!////////////////////////////////////////////////////!

module funcs
  implicit none
  double precision, parameter :: pi = 3.1415926535897932384626433832795d0

  private pi

contains


function factorial(n)  result(f)
  implicit none
  integer, intent(in) :: n
  integer :: i
  double precision :: f

  f = 1d0
  do i = 1, n
    f = f*dble(i)
  end do

end function factorial


!///////////////////!
! Special Functions
! - Hermite polynomials
! - Error Function
! - Gamma Function

function Her(k,x) !Hermite polynomials
  implicit none
  !I/O
  integer, intent(in) :: k
  double precision, intent(in) :: x
  !internal
  integer :: i
  double precision :: Her, y0, y1, y2

  Her = 0d0
  if(k==-1) Her = dsqrt(pi*0.5d0)*dexp(x**2*0.5d0)*erfc(x/dsqrt(2.d0))
  y0 = 1d0
  if(k==0) Her = y0
  y1 = x
  if(k==1) Her = y1
  if(k>=2) then
    do i = 2, k
      y2 = x*y1-dble(i-1)*y0
      y0 = y1
      y1 = y2
    end do
    Her = y1
  end if

end function Her


function erfc(x)
  implicit none
  double precision, intent(in) :: x
  double precision erfc

  if (x < 0) then
    erfc = 1.0+GammaFuncP(0.5d0,x*x)
  else
    erfc = GammaFuncQ(0.5d0,x*x)
  end if

end function erfc


function GammaFuncP(a,x)
  implicit none
  double precision, intent(in) :: a,x
  double precision gamser, gammcf, gln, GammaFuncP

  if (x < (a+1.d0)) then
    call gser(gamser,a,x,gln)
    GammaFuncP = gamser
  else
    call gcf(gammcf,a,x,gln)
    GammafuncP = 1.d0 - gammcf
  end if

end function GammaFuncP


function GammaFuncQ(a,x)
  implicit none
  double precision, intent(in) :: a,x
  double precision gamser, gammcf, gln, GammaFuncQ

  if (x < (a+1d0)) then
    call gser(gamser,a,x,gln)
    GammaFuncQ = 1d0-gamser
  else
    call gcf(gammcf,a,x,gln)
    GammafuncQ = gammcf
  end if

end function GammaFuncQ


subroutine gser(gamser,a,x,gln)
  implicit none
  double precision, intent(in) :: a,x
  double precision, intent(out) :: gamser, gln
  double precision :: EPS, ap, sum, del
  integer :: ITMAX, n

  ITMAX = 100
  EPS   = 3d-7 !numerical error
  gln   = lnGamma(a)

  if (x<=0d0) then
    if (x < 0) write(*,*) 'x less than 0 in routine gser'
    gamser = 0d0
  else
    ap = a
    sum = 1d0/a
    del = sum
    do n = 1, ITMAX
      ap = ap + 1d0
      del = del*(x/ap)
      sum = sum + del
      if (abs(del)<abs(sum)*EPS) gamser = sum*dexp(-x+a*dlog(x)-gln)
    end do
  end if

end subroutine gser


subroutine gcf(gammcf,a,x,gln)
  implicit none
  double precision, intent(in) :: a,x
  double precision, intent(out) :: gammcf, gln
  double precision :: EPS, FPMIN, gold, fac, g, b1, b0, anf, ana, an, a1, a0
  integer :: ITMAX, n

  ITMAX = 100
  EPS   = 3d-7 !numerical error
  b0    = 0d0
  gold  = 0d0
  fac   = 1d0
  b1    = 1d0
  a0    = 1d0
  gln   = lnGamma(a)

  a1 = x
  do n = 1, ITMAX
    an = dble(n)
    ana = an -a
    a0 = (a1+a0*ana)*fac
    b0 = (b1+b0*ana)*fac
    anf = an*fac
    a1 = x*a0+anf*a1
    b1 = x*b0+anf*b1
    if (.not.a1==0) then
      fac = 1d0/a1
      g = b1*fac
      if (abs(g-gold)/g < EPS) then
        gammcf = dexp(-x+a*dlog(x)-gln)*g
        exit
      end if
      gold = g
    end if
  end do

end subroutine gcf


function lnGamma(x) !Lanczos formula (gamma=5, N=6)
  implicit none
  double precision, intent(in) :: x
  double precision tmp, ser, lnGamma
  double precision :: cof(6), cof_0
  integer j

  cof_0  = 1.000000000190015d0
  cof(1) = 76.18009172947146d0
  cof(2) = -86.50532032941677d0
  cof(3) = 24.01409824083091d0
  cof(4) = -1.231739572450155d0
  cof(5) = 0.1208650973866179d-2
  cof(6) = -0.5395239384953d-5

  tmp = (x+0.5d0)*dlog(x+5.5d0) - (x+5.5d0)
  ser = cof_0
  do j = 1, 6
    ser = ser + cof(j)/(x+j)
  end do
  lnGamma = tmp + dlog(2.5066282746310005d0*ser/x) !ln[Gammma(x+1)]-ln(x)

end function lnGamma


!//// Bessel functions ////!

function Bessel_J(n,x,eps)  result(f)
! Bessel function
  implicit none
  !I/O
  integer, intent(in) :: n
  integer, intent(in), optional :: eps
  double precision, intent(in) :: x
  !internal
  integer :: i, m
  double precision :: f, dt, t

  m = 32
  if(present(eps)) m = eps

  t = 0d0
  dt = 2d0*pi/dble(m)
  f = 0d0
  do i = 1, m-1
    t = t + dt
    f = f + dcos(x*dsin(t)-n*t)*dt/(2d0*pi)
  end do
  f = f + 1d0*dt/(2d0*pi)

end function Bessel_J



!//// HyperGeometric Function ////!

subroutine hygfx(a,b,c,x,hf)
! //// Hypergeometric function F(A,B,C,X) //// !
! * Licensing:
!    The original FORTRAN77 version of this routine is copyrighted by
!    Shanjie Zhang and Jianming Jin.
!
! * Modified:
!    08 September 2007
!
  implicit none
  integer(kind=4) :: j,k,n,nm,m
  logical :: l0,l1,l2,l3,l4,l5
  double precision :: a,a0,aa,b,bb,c,c0,c1,eps,f0,f1,g0,g1,g2,g3,ga,gabc,gam,sp,x1
  double precision :: gb,gbm,gc,gca,gcab,gcb,gm,hf,hw,pa,pb,r,r0,r1,rm,rp,sm,sp0,x
  double precision, parameter :: el = 0.5772156649015329

  l0 = ( c == aint(c) ) .and. ( c < 0d0 )
  l1 = ( 1d0 - x < 1.d-15 ) .and. ( c-a-b <= 0d0 )
  l2 = ( a == aint(a) ) .and. ( a < 0d0 )
  l3 = ( b == aint(b) ) .and. ( b < 0d0 )
  l4 = ( c - a == aint(c-a) ) .and. ( c-a <= 0d0 )
  l5 = ( c - b == aint(c-b) ) .and. ( c-b <= 0d0 )

  if ( l0 .or. l1 ) then
    write ( *, '(a)' ) ' '
    write ( *, '(a)' ) 'HYGFX - Fatal error!'
    write ( *, '(a)' ) '  The hypergeometric series is divergent.'
    return
  end if

  if ( 0.95D0 < x ) then
    eps = 1.0D-8
  else
    eps = 1.0D-15
  end if

  if ( x == 0d0 .or. a == 0d0 .or. b == 0d0 ) then
    hf = 1d0
    return
  else if ( 1d0-x == eps .and. 0d0 < c-a-b) then
    call gamma(c,gc)
    call gamma(c-a-b,gcab)
    call gamma(c-a,gca)
    call gamma(c-b,gcb)
    hf = gc*gcab/(gca*gcb)
    return
  else if (1d0+x<=eps .and. abs(c-a+b-1d0)<=eps) then
    g0 = sqrt(pi)*2d0**(-a)
    call gamma(c,g1)
    call gamma(1d0 + a/2d0 - b, g2 )
    call gamma(0.5d0 + 0.5d0*a, g3 )
    hf = g0*g1/(g2*g3)
    return
  else if ( l2 .or. l3 ) then
    if(l2) nm = int(abs(a))
    if(l3) nm = int(abs(b))
    hf = 1d0
    r  = 1d0
    do k = 1, nm
      r  = r*(a+k-1d0)*(b+k-1d0)/(k*(c+k-1d0))*x
      hf = hf + r
    end do
    return
  else if ( l4 .or. l5 ) then
    if(l4) nm = int(abs(c-a))
    if(l5) nm = int(abs(c-b))
    hf = 1d0
    r  = 1d0
    do k = 1, nm
      r = r*(c-a+k-1d0)*(c-b+k-1d0)/(k*(c+k-1d0))*x
      hf = hf + r
    end do
    hf = (1d0-x)**(c-a-b)*hf
    return
  end if
  aa = a
  bb = b
  x1 = x
!
!  WARNING: ALTERATION OF INPUT ARGUMENTS A AND B, WHICH MIGHT BE CONSTANTS.
!
  if ( x < 0d0 ) then
    x = x/(x-1d0)
    if ( a < c .and. b < a .and. 0d0 < b ) then
      a = bb
      b = aa
    end if
    b = c - b
  end if
  if ( 0.75D0 <= x ) then
    gm = 0d0
    if ( abs(c-a-b-aint(c-a-b)) < 1d-15 ) then
      m = int(c-a-b)
      call gamma(a,ga)
      call gamma(b,gb)
      call gamma(c,gc)
      call gamma(a+m,gam)
      call gamma(b+m,gbm)
      call psi(a,pa)
      call psi(b,pb)
      if (m/= 0) gm = 1d0
      do j = 1, abs(m)-1
        gm = gm*j
      end do
      rm = 1d0
      do j = 1, abs(m)
        rm = rm * j
      end do
      f0 = 1d0
      r0 = 1d0
      r1 = 1d0
      sp0 = 0d0
      sp = 0d0
      if ( 0 <= m ) then
        c0 = gm * gc / ( gam * gbm )
        c1 = - gc * ( x - 1d0 )**m / ( ga * gb * rm )
        do k = 1, m - 1
          r0 = r0*(a+k-1d0)*(b+k-1d0)/(k*(k-m))*(1d0-x)
          f0 = f0 + r0
        end do
        do k = 1, m
          sp0 = sp0+1d0/(a+k-1d0)+1d0/(b+k-1d0)-1d0/dble(k)
        end do
        f1 = pa + pb + sp0 + 2d0*el + log(1d0-x)
        hw = f1
        do k = 1, 250
          sp = sp + (1d0-a)/(k*(a+k-1d0))+(1d0-b)/(k*(b+k-1d0))
          sm = 0d0
          do j = 1, m
            sm = sm+(1d0-a)/((j+k)*(a+j+k-1d0))+1d0/(b+j+k-1d0)
          end do
          rp = pa + pb + 2d0*el + sp + sm + log(1d0-x)
          r1 = r1*(a+m+k-1d0)*(b+m+k-1d0)/(k*(m+k))*(1d0-x)
          f1 = f1 + r1 * rp
          if ( abs(f1-hw) < abs(f1)*eps )  exit
          hw = f1
        end do
        hf = f0 * c0 + f1 * c1
      else if ( m < 0 ) then
        m = - m
        c0 = gm * gc / ( ga * gb * ( 1d0 - x )**m )
        c1 = - ( - 1 )**m * gc / ( gam * gbm * rm )
        do k = 1, m - 1
          r0 = r0*(a-m+k-1d0)*(b-m+k-1d0)/(k*(k-m))*(1d0-x)
          f0 = f0 + r0
        end do
        do k = 1, m
          sp0 = sp0 + 1d0/dble(k)
        end do
        f1 = pa + pb - sp0 + 2d0*el + log(1d0-x)
        do k = 1, 250
          sp = sp + (1d0-a)/(k*(a+k-1d0)) + (1d0-b)/(k*(b+k-1d0))
          sm = 0d0
          do j = 1, m
            sm = sm + 1d0/dble(j+k)
          end do
          rp = pa+pb+2d0*el+sp-sm+log(1d0-x)
          r1 = r1*(a+k-1d0)*(b+k-1d0)/(k*(m+k))*(1d0-x)
          f1 = f1+r1*rp
          if ( abs(f1-hw) < abs(f1)*eps )  exit
          hw = f1
        end do
        hf = f0 * c0 + f1 * c1
      end if
    else
      call gamma(a,ga)
      call gamma(b,gb)
      call gamma(c,gc)
      call gamma(c-a,gca)
      call gamma(c-b,gcb)
      call gamma(c-a-b,gcab)
      call gamma(a+b-c,gabc)
      c0 = gc*gcab/(gca*gcb)
      c1 = gc*gabc/(ga*gb) * (1d0-x)**(c-a-b)
      hf = 0d0
      r0 = c0
      r1 = c1
      hw = hf !namikawa
      do k = 1, 250
        r0 = r0*(a+k-1d0)*(b+k-1d0)/(k*(a+b-c+k))*(1d0-x)
        r1 = r1*(c-a+k-1d0)*(c-b+k-1d0)/(k*(c-a-b+k))*(1d0-x)
        hf = hf + r0 + r1
        if ( abs(hf-hw) < abs(hf)*eps )   exit
        hw = hf
      end do
      hf = hf + c0 + c1
    end if
  else
    a0 = 1d0
    if (a<c .and. c<2d0*a .and. b<c .and. c<2d0*b ) then
      a0 = (1d0-x)**(c-a-b)
      a = c-a
      b = c-b
    end if
    hf = 1d0
    r  = 1d0
    hw = hf !namikawa
    do k = 1, 250
      r = r*(a+k-1d0)*(b+k-1d0)/(k*(c+k-1d0))*x
      hf = hf+r
      if ( abs(hf-hw) <= abs(hf)*eps )  exit
      hw = hf
    end do
    hf = a0 * hf
  end if
  if ( x1 < 0d0 ) then
    x = x1
    c0 = 1d0 / ( 1d0 - x )**aa
    hf = c0 * hf
  end if
  a = aa
  b = bb
  if ( 120 < k ) then
    write ( *, '(a)' ) ' '
    write ( *, '(a)' ) 'HYGFX - Warning!'
    write ( *, '(a)' ) '  A large number of iterations were needed.'
    write ( *, '(a)' ) '  The accuracy of the results should be checked.'
  end if
  return

end subroutine hygfx


subroutine gamma(x,ga)
! //// Gamma function //// !
! * Licensing:
!    The original FORTRAN77 version of this routine is copyrighted by
!    Shanjie Zhang and Jianming Jin.
! * Modified:
!    08 September 2007
  implicit none
  double precision, dimension(26) :: g = (/1.0D0, 0.5772156649015329D0, &
   -0.6558780715202538D0, -0.420026350340952D-1, 0.1665386113822915D0, &
   -0.421977345555443D-1, -0.96219715278770D-2, 0.72189432466630D-2, &
   -0.11651675918591D-2, -0.2152416741149D-3, 0.1280502823882D-3, &
   -0.201348547807D-4, -0.12504934821D-5, 0.11330272320D-5, &
   -0.2056338417D-6, 0.61160950D-8, 0.50020075D-8, -0.11812746D-8, &
    0.1043427D-9, 0.77823D-11, -0.36968D-11, 0.51D-12, &
   -0.206D-13, -0.54D-14, 0.14D-14, 0.1D-15 /)
  integer(kind=4) :: k,m,m1
  double precision :: r,x,z,ga,gr

  if (x == aint(x)) then
    if (0.0D0 < x) then
      ga = 1.0D0
      m1 = int(x) - 1
      do k = 2, m1
        ga = ga*k
      end do
    else
      ga = 1.0D+300
    end if
  else
    if ( 1.0D0 < abs(x) ) then
      z = abs(x)
      m = int(z)
      r = 1.0D0
      do k = 1, m
        r = r*(z-real(k,kind=8))
      end do
      z = z-real(m,kind=8)
    else
      z = x
    end if
    gr = g(26)
    do k = 25, 1, -1
      gr = gr*z + g(k)
    end do
    ga = 1.0D0/(gr*z)
    if ( 1.0D0 < abs(x) ) then
      ga = ga*r
      if ( x < 0.0D0 ) then
        ga = -pi/(x*ga*sin(pi*x))
      end if
    end if
  end if
  return

end subroutine gamma


subroutine psi(x,ps)
! //// PSI function //// !
! * Licensing:
!    The original FORTRAN77 version of this routine is copyrighted by
!    Shanjie Zhang and Jianming Jin.
!
!  Modified:
!    08 September 2007
!
  implicit none
  integer(kind=4) :: k, n
  double precision :: ps,s,x,x2,xa
  double precision, parameter :: a1 = -0.083333333333333333
  double precision, parameter :: a2 =  0.0083333333333333333
  double precision, parameter :: a3 = -0.0039682539682539683
  double precision, parameter :: a4 =  0.0041666666666666667
  double precision, parameter :: a5 = -0.0075757575757575758
  double precision, parameter :: a6 =  0.021092796092796093
  double precision, parameter :: a7 = -0.083333333333333333
  double precision, parameter :: a8 =  0.4432598039215686
  double precision, parameter :: el = 0.5772156649015329

  xa = abs(x)
  s = 0.0D0

  if ( x == aint(x) .and. x <= 0.0D0 ) then
    ps = 1.0D+300
    return
  else if ( xa == aint(xa) ) then
    n = int (xa)
    do k = 1, n - 1
      s = s + 1.0D0/real(k,kind=8)
    end do
    ps = - el + s
  else if ( xa + 0.5D0 == aint(xa+0.5D0) ) then
    n = int(xa-0.5D0)
    do k = 1, n
      s = s + 1.0D0 / real(2*k-1,kind=8)
    end do
    ps = - el + 2.0D0 * s - 1.386294361119891D0
  else
    if ( xa < 10.0D0 ) then
      n = 10 - int(xa)
      do k = 0, n - 1
        s = s + 1.0D0 / ( xa + real(k,kind=8) )
      end do
      xa = xa + real(n,kind=8)
    end if
    x2 = 1.0D0/(xa*xa)
    ps = log(xa) - 0.5D0/xa + x2*((((((( a8 * x2 + a7 ) * x2 + a6 ) &
      * x2 + a5 ) * x2 + a4 ) * x2 + a3 ) * x2 + a2 ) * x2 + a1 )
    ps = ps - s
  end if
  if ( x < 0.0D0 ) then
    ps = ps - pi * cos ( pi * x ) / sin ( pi * x ) - 1.0D0 / x
  end if
  return

end subroutine psi


end module funcs
