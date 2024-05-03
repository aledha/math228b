using SparseArrays, Polynomials, PyCall, LinearAlgebra
using Plots
default(legend=false, linewidth=2)

"""
    mkanim(x, allu; [filename, axis])

Utility for animating PDE solutions.
"""
function mkanim(x, allu; filename=tempname() * ".mp4", axis=[0,1,-0.1,1.1])
    anim = @animate for i in 1:length(allu)
        plot(xlims=axis[1:2], ylims=axis[3:4])
        plot!(x[1], allu[i][1], linecolor=:blue)
        plot!(x[2], allu[i][2], linecolor=:black)
    end
    gif(anim, filename)
end

"""
    function gauss_quad(p)

Gaussian quadrature on [-1,1] for given degree of precision `p`
"""
function gauss_quad(p)
    n = ceil((p+1)/2)
    b = 1:n-1
    b = @. b / sqrt(4*b^2 - 1)
    eval, evec = eigen(diagm(1 => b, -1 => b))
    return eval, 2*evec[1,:].^2
end

"""
    function legendre_poly(x, p)

Legendre polynomials and derivatives up to degree `p` at nodes `x`
"""
function legendre_poly(x, p)
    z = zeros(size(x))
    o = ones(size(x))
    y = hcat(o, x, repeat(z, 1, p-1))
    dy = hcat(z, o, repeat(z, 1, p-1))
    for i = 1:p-1
        @. y[:,i+2] = ((2i+1)*x*y[:,i+1] - i*y[:,i]) / (i+1)
        @. dy[:,i+2] = ((2i+1)*(x*dy[:,i+1] + y[:,i+1]) - i*dy[:,i]) / (i+1)
    end
    y, dy
end

function dgconvect(; n=10, p=1, T=1.0, dt=1e-3)
    # Discretization
    h = 1 / n
    #s = @. (0:p) / p
    s0 = [-cos(pi*i/p) for i=0:p] 
    s = @. (s0 + 1) * h/2 
    x = @. s + (0:h:1-h)'

    # Gaussian initial condition (and exact solution if shifted)
    uinit(x) = exp(-(x - 0.5)^2 / 0.1^2)

    yy, dyy = legendre_poly(s0, p)
    C = inv(yy)
    
    # Evaluate Legendre polynomials at Gaussian nodes
    gx, gw = gauss_quad(2*p)
    gyy, gdyy = legendre_poly(gx, p)

    # Basis function number i evaluated at Gaussian nodes
    gphi = gyy * C
    # Scale derivative 
    gphider = 2/h .* gdyy * C

    Mel = h/2 .* [dot(gphi[:,i] .* gphi[:,j], gw) for i=1:p+1, j=1:p+1]
    Kel = h/2 .* [dot(gphider[:,i] .* gphi[:,j], gw) for i=1:p+1, j=1:p+1]


    # RHS function in semi-discrete system
    function rhs(u)
        r = Kel * u
        r[end,:] = r[end,:] - u[end,:]
        r[1,:] = r[1,:] + u[end, [end; 1:end-1]]
        r = Mel \ r
    end

    # Setup
    u = uinit.(x)
    nsteps = round(Int, T/dt)

    # Setup plotting
    x2 = @. h*(0:1/(3*p-1):1) + (0:h:1-h)'
    x2 = reduce(vcat, x2)

    uplot = uinit.(x2)
    xplot = -1:2/(3p-1):1
    yyplot, dyyplot = legendre_poly(xplot, p)
    basis = yyplot*C
    
    xx = (0:0.01:1) # For exact solution
    allu = [ (uplot, uinit.(xx)) ]

    # Main loop
    for it = 1:nsteps
        # Runge-Kutta 4
        k1 = dt * rhs(u)
        k2 = dt * rhs(u + k1/2)
        k3 = dt * rhs(u + k2/2)
        k4 = dt * rhs(u + k3)
        u += (k1 + 2*k2 + 2*k3 + k4) / 6

        # Plotting
        if mod(it, round(nsteps/100)) == 0
            uexact = @. uinit(mod(xx - dt*it, 1.0))
            uplot = basis * u
            uplot = reduce(vcat, uplot)
            push!(allu, (uplot, uexact))
        end
    end

    #uexact = @. uinit(mod(x - T, 1.0))  # Exact final solution
    #error = maximum(abs.(u - uexact))   # Discrete inf-norm error""

    gu = gphi*u                             # Numerical solution evaluated at Gaussian nodes in each element
    gxx = @. h/2*(gx+1) + (0:h:1-h)'        # Gaussian nodes in each element
    guexact = @. uinit(mod(gxx - T, 1.0))   # Exact final solution evaluated at Gaussian nodes in each element

    difference = @. (gu-guexact)^2

    error = sqrt(sum(gw' * difference))

    return (x2,xx), allu, error
end

function dgconvect_convergence()
    ps = [1,2,4,8,16]
    dt = 2 * 10^-4
    T = 1
    np = [16,32,64,128,256]

    errors = zeros(5,5)
    slopes = zeros(5)
    nps = zeros(5,5)
    
    convplot = plot(legend=true, xlabel="np", ylabel="error")
    for i = 1:5
        p = ps[i]
        for j = 1:5
            n = Int64(np[j]/p)
            x, allu, error = dgconvect(p=p, n=n, dt=dt, T=T)
            errors[i,j] = error
            nps[i,j] = n*p
        end
        if i < 5
            fit = Polynomials.fit(log.(nps[i,:]), log.(errors[i,:]),1)
            slopes[i] = coeffs(fit)[2]
            plot!(nps[i,:], errors[i,:], xaxis=:log, yaxis=:log, labels="p=$(ps[i]), slope = $(round(slopes[i], digits=2))")
        else
            # Throw away last point due to rounding errors
            fit = Polynomials.fit(log.(nps[5,begin+1:end-2]), log.(errors[5,begin+1:end-2]), 1)
            slopes[5] = coeffs(fit)[2]
            plot!(nps[5,begin:end], errors[5,begin:end], xaxis=:log, yaxis=:log, labels="p=$(ps[5]), slope = $(round(slopes[5], digits=2))")
        end
    end
  
    display(convplot)

    return errors, slopes
end

function dgconvdiff(; n=10, p=1, T=1.0, dt=1e-3,k=1e-3)

    # Discretization
    h = 1 / n
    #s = @. (0:p) / p
    s0 = [-cos(pi*i/p) for i=0:p] 
    s = @. (s0 + 1) * h/2 
    x = @. s + (0:h:1-h)'
    
    u_exact(x,t) = sum([1/sqrt(1+400*k*t) * exp(-100 * (x - 0.5 - t + i)^2 / (1+400*k*t)) for i=-2:2])

    yy, dyy = legendre_poly(s0, p)
    C = inv(yy)
    
    # Evaluate Legendre polynomials at Gaussian nodes
    gx, gw = gauss_quad(2*p)
    gyy, gdyy = legendre_poly(gx, p)

    
    # Basis function number i evaluated at Gaussian nodes
    gphi = gyy * C
    # Scale derivative 
    gphider = 2/h .* gdyy * C

    Mel = h/2 .* [dot(gphi[:,i] .* gphi[:,j], gw) for i=1:p+1, j=1:p+1]
    Kel = h/2 .* [dot(gphider[:,i] .* gphi[:,j], gw) for i=1:p+1, j=1:p+1]

    # RHS function in semi-discrete system
    function rhs(u)
        sigmar = -Kel * u
        sigmar[1,:] = sigmar[1,:] - u[1,:]
        sigmar[end,:] = sigmar[end,:] + u[1, [2:end; 1]]
        sigma = Mel \ sigmar

        r = Kel * (u - k .* sigma)
        r[end,:] = r[end,:] - (u[end,:] - k .* sigma[end,:])
        r[1,:] = r[1,:] + (u[end, [end; 1:end-1]] - k .* sigma[end, [end; 1:end-1]])
        r = Mel \ r
    end

    # Setup
    u = u_exact.(x,0)
    nsteps = round(Int, T/dt)
    
    # Setup plotting
    x2 = @. h*(0:1/(3*p-1):1) + (0:h:1-h)'
    x2 = reduce(vcat, x2)

    uplot = u_exact.(x2,0)
    xplot = -1:2/(3p-1):1
    yyplot, dyyplot = legendre_poly(xplot, p)
    basis = yyplot*C
    
    xx = (0:0.01:1) # For exact solution
    allu = [ (uplot, u_exact.(xx,0)) ]

    # Main loop
    for it = 1:nsteps
        # Runge-Kutta 4
        k1 = dt * rhs(u)
        k2 = dt * rhs(u + k1/2)
        k3 = dt * rhs(u + k2/2)
        k4 = dt * rhs(u + k3)
        u += (k1 + 2*k2 + 2*k3 + k4) / 6

        # Plotting
        if mod(it, round(nsteps/100)) == 0
            uexact = @. u_exact(xx, dt*it)
            uplot = basis * u
            uplot = reduce(vcat, uplot)
            push!(allu, (uplot, uexact))
        end
    end
    
    gu = gphi*u                             # Numerical solution evaluated at Gaussian nodes in each element
    gxx = @. h/2*(gx+1) + (0:h:1-h)'        # Gaussian nodes in each element
    guexact = @. u_exact(gxx, T)   # Exact final solution evaluated at Gaussian nodes in each element

    difference = @. (gu-guexact)^2

    error = sqrt(sum(gw' * difference))
  
    return (x2,xx), allu, error
end

function dgconvdiff_convergence()
    ps = [1,2,4,8,16]
    dt = 2 * 10^-4
    T = 1
    np = [16,32,64,128,256]

    errors = zeros(5,5)
    slopes = zeros(5)
    nps = zeros(5,5)
    
    convplot = plot(legend=true, xlabel="np", ylabel="error")
    for i = 1:5
        p = ps[i]
        for j = 1:5
            n = Int64(np[j]/p)
            x, allu, error = dgconvdiff(p=p, n=n, dt=dt, T=T)
            errors[i,j] = error
            nps[i,j] = n*p
        end
        if i < 5
            fit = Polynomials.fit(log.(nps[i,:]), log.(errors[i,:]),1)
            slopes[i] = coeffs(fit)[2]
            plot!(nps[i,:], errors[i,:], xaxis=:log, yaxis=:log, labels="p=$(ps[i]), slope = $(round(slopes[i], digits=2))")
        else
            # Throw away last points due to rounding errors
            fit = Polynomials.fit(log.(nps[5,begin+1:end-2]), log.(errors[5,begin+1:end-2]), 1)
            slopes[5] = coeffs(fit)[2]
            plot!(nps[5,begin:end], errors[5,begin:end], xaxis=:log, yaxis=:log, labels="p=$(ps[5]), slope = $(round(slopes[5], digits=2))")
        end
    end
  
    display(convplot)

    return errors, slopes 
end

#errors, slopes = dgconvdiff_convergence()