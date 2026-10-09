!***************************************************************************
!*                                                                         *
!*             1-D Shallow-Water equations semi-implicit solver            *
!*                                                                         *
!*                           Kai Logemann, HZG, October 2020               *
!*                                                                         *
!***************************************************************************

program shallow_water
implicit none

integer i,nx,ip,im
integer :: step, nsteps, output_steps
character(256) :: argument
real :: duration, save_interval, initial_center
allocatable :: u,un,zeta,zetan,h,uint,us,uints,div,ce,cw,zs,d

real dt,dx,u(:),un(:),zeta(:),zetan(:),h(:),cd,g,uint(:),hm,us(:),uints(:), &
    & div(:)
real wimp,ce(:),cw(:),zetnew,zs(:),dif,maxdif,d(:)
real x, pi, sigma, mu, uu, adv, AH, adv_coefficient
integer it


! set constants ==================================

call get_command_argument(1, argument)
read(argument,*) dx
call get_command_argument(2, argument)
read(argument,*) dt
call get_command_argument(3, argument)
read(argument,*) AH
call get_command_argument(4, argument)
read(argument,*) duration
call get_command_argument(5, argument)
read(argument,*) save_interval
call get_command_argument(6, argument)
read(argument,*) initial_center
nx = nint(2000000./dx) + 2
nsteps = nint(duration/dt)
output_steps = nint(save_interval/dt)
allocate(u(nx),un(nx),zeta(nx),zetan(nx),h(nx),uint(nx),us(nx), &
 uints(nx),div(nx),ce(nx),cw(nx),zs(nx),d(nx))
open(70,file='u.bin',access='stream',form='unformatted',status='replace')
open(71,file='zeta.bin',access='stream',form='unformatted',status='replace')
h = 0.

do i = 2,nx-1,1
 d(i) = 100. ! undisturbed water depth [m], topography
enddo
d(1) = -10. ! dry boundary cell
d(nx) = -10. ! dry boundary cell


adv_coefficient = 1. ! coefficient for in-/excluding advection
cd = 0.0 ! 2.e-3 ! bottom drag coefficient
! AH is supplied in dimensional m**2/s.
g = 9.81 ! acceleration due to gravity [m/s**2]

uint = 0. ! vertically integrated velocity [m**2/s]

wimp = 0.5 ! weight between time levels

PI  = 4. * ATAN(1.)

! initalisation ====================================

u = 0. ! velocity [m/s]
un = 0. ! velocity at new time step [m/s]
us = 0. ! interim solution of velocity [m/s]

zeta = 0. ! sea surface elevation [m]
zetan = 0. ! sea surface elevation at new time level [m]

do i = 2,nx-1,1

  x = (REAL(i)-1.5) * dx - 1000000. ! wet cell centers [m]

  mu = initial_center
  sigma = 100000.0
  zeta(i) = 2.5 * sigma * (1. / (sigma * sqrt(2*pi))) * exp(-0.5 * ( (x-mu) / sigma &
    & )**2. )

  h(i) = d(i) + zeta(i)

enddo

! ======================================================

call write_snapshot()

! ======================================================

do step = 1,nsteps

! equation of motion => us (interim solution)
do i = 1,nx,1
  im = max(1,i-1)
  ip = min(nx,i+1)

  ! momentum advection (upstream) => adv -----------------
    uu = 0.5*u(i)+0.25*u(im)+0.25*u(ip)

    if(uu.ge.0.) then
      uu = max(0.,u(im))
      adv = -dt*uu*(u(i)-u(im))/dx
    else
      uu = min(0.,u(ip))
      adv = -dt*uu*(u(ip)-u(i))/dx
    endif

  if((d(i).gt.0.1).and.(d(ip).gt.0.1)) then
    hm = 0.5*(h(i)+h(ip))
    us(i) = u(i)- dt*cd/hm*u(i)*abs(u(i)) - (1.-wimp)*dt*g*(zeta(ip)-zeta(i))/dx &
            + dt * AH * (u(i+1) - 2. * u(i) + u(i-1)) / (dx**2.) &
            + adv * adv_coefficient
  else
    us(i) = 0.
  endif
enddo

! vertical integrals of velocity => uint, and interim solution uints

do i = 1,nx,1
  im = max(1,i-1)
  ip = min(nx,i+1)
  if(d(i).gt.0.0.and.(d(ip).gt.0.0)) then
    hm = 0.5*(h(i)+h(ip))
    uint(i) = hm*u(i)
    uints(i) = hm*us(i)
  else
    uint(i) = 0.
    uints(i) = 0.
  endif
enddo

! equation of continuity, first compute div

div = 0.
do i = 1,nx,1
  if(d(i).gt.0.0) then
    im = max(1,i-1)
    div(i) = -dt*(1.-wimp)*(uint(i)-uint(im))/dx  - dt*wimp*(uints(i)-uints(im))/dx
  endif
enddo

! now compute coefficients for zetan system of equations: ce and cw

ce = 0.
cw = 0.

do i = 1,nx,1
  im = max(1,i-1)
  ip = min(nx,i+1)

  if((d(i).gt.0.0).and.(d(ip).gt.0.0)) then
    ce(i) = dt*dt*wimp*wimp*g*0.5*(h(i)+h(ip))/(dx*dx)
  endif

  if((d(i).gt.0.0).and.(d(im).gt.0.0)) then
    cw(i) = dt*dt*wimp*wimp*g*0.5*(h(i)+h(im))/(dx*dx)
  endif

enddo

! iterative solution of the system of eqations

it = 0

zs = zeta

1 continue
maxdif = 0.
it = it+1

do i = 1,nx,1
  im = max(1,i-1)
  ip = min(nx,i+1)
  if(d(i).gt.0.1) then
    zetnew = 1./(1. + ce(i) + cw(i))*(zeta(i) + div(i) + ce(i)*zetan(ip) + &
    & cw(i)*zetan(im))

    dif = abs(zetnew-zs(i))
    maxdif = max(maxdif,dif)
    zs(i) = zetnew
  else
    zs(i) = 0.
  endif
enddo

zetan = zs ! update iterative solution (could be also done in the loop above, but &
! would lead to problems when parallelisation is applied.)

!write(*,*) 'it,maxdif',it,maxdif
!read(*,*)

if(it.gt.100000) error stop 'Iteration failed to converge'
if(maxdif.gt.1.e-6) goto 1 ! convergence check

! ===============================================================
! Compute velocity at the new time level.

do i = 1,nx,1
  im = max(1,i-1)
  ip = min(nx,i+1)
  if((d(i).gt.0.1).and.(d(ip).gt.0.1)) then
    un(i) = us(i) - wimp*dt*g*(zetan(ip)-zetan(i))/dx
  else
    un(i) = 0.
  endif
enddo

! ================================================================
! swap time level

u = un
zeta = zetan

do i = 1,nx,1
  if(d(i).gt.0.1) then
    h(i) = d(i) + zeta(i)
  endif
enddo

! ================================================
! Store the staggered wet grid only, at exact requested timesteps.
if(mod(step,output_steps).eq.0) call write_snapshot()
enddo
close(70)
close(71)

contains



! ========================================================================

  subroutine write_snapshot()
  write(70) u(1:nx-1)
  write(71) zeta(2:nx-1)
  end subroutine write_snapshot

end program shallow_water
