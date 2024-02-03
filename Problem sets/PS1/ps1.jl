using SparseArrays, PyPlot, Polynomials

# Solve Poisson's equation -(uxx + uyy) = f, bnd cnds u(x,y) = g(x,y)
# on a square grid using the finite difference method.
#
# UC Berkeley Math 228B, Per-Olof Persson <persson@berkeley.edu>

using SparseArrays, Plots


function assemblePoisson(n, f, g)
    h = 1.0 / n
    N = (n+1)^2
    x = h * (0:n)
    y = x

    umap = reshape(1:N, n+1, n+1)     # Index mapping from 2D grid to vector
    A = Tuple{Int64,Int64,Float64}[]  # Array of matrix elements (row,col,value)
    b = zeros(N)

    # Main loop, insert stencil in matrix for each node point
    for j = 1:n+1
        for i = 1:n+1
            row = umap[i,j]
            if i == 1 || i == n+1 || j == 1 || j == n+1
                # Dirichlet boundary condition, u = g
                push!(A, (row, row, 1.0))
                b[row] = g(x[i],y[j])
            else
                # Interior nodes, 9-point stencil
                push!(A, (row, row, -20.0))
                push!(A, (row, umap[i+1,j], 4.0))
                push!(A, (row, umap[i-1,j], 4.0))
                push!(A, (row, umap[i,j+1], 4.0))
                push!(A, (row, umap[i,j-1], 4.0))
                push!(A, (row, umap[i-1,j-1], 1.0))
                push!(A, (row, umap[i-1,j+1], 1.0))
                push!(A, (row, umap[i+1,j-1], 1.0))
                push!(A, (row, umap[i+1,j+1], 1.0))
 
                b[row] = -6*h^2 * (f(x[i], y[j]) + 1/12 * (f(x[i+1],y[j]) + f(x[i-1],y[j]) +
                f(x[i],y[j+1]) + f(x[i],y[j-1]) - 4*f(x[i],y[j])))

            end
        end
    end

    # Create CSC sparse matrix from matrix elements
    A = sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), N, N)

    return A, b, x, y
end

function testPoisson(n=40)
    uexact(x,y) = exp(-(4(x - 0.3)^2 + 9(y - 0.6)^2))
    f(x,y) = uexact(x,y) * (26 - (18y - 10.8)^2 - (8x - 2.4)^2)
    A, b, x, y = assemblePoisson(n, f, uexact)

    # Solve + reshape solution into grid array
    u = reshape(A \ b, n+1, n+1)
 
    # Compute error in max-norm
    u0 = uexact.(x, y')
    error = maximum(abs.(u - u0))
end

function problem1b(ns)
    errors = zeros(length(ns))
    hs = zeros(length(ns))
    for i in (1:length(ns))
        errors[i] = testPoisson(trunc(Int64,ns[i]))
        hs[i] = 1 / ns[i]
    end
    fit = Polynomials.fit(log.(hs), log.(errors), 1)
    order = coeffs(fit)[2]

    fig, ax = subplots()
    ax.loglog(hs, errors, label= "Order of convergence: $(order)")
    ax.set_title("Convergence plot")
    ax.set_xlabel("h")
    ax.set_ylabel("error")
    ax.legend()
    display(fig)
end

ns = [300, 200, 150, 100, 50, 20, 10]
#problem1b(ns)



function channelFlow(L, B, H, n)
    
    A = (1/4 * (L-B)^2 - H^2)^(1/2)
    h = 1.0 / n
    N = (n+1)^2
    xi = h * (0:n)
    eta = xi

    J(i, j) = B*H/2 + A*H*eta[j]
    a(i, j) = (A*xi[i])^2 + H^2
    b(i, j) = (B/2 + A*eta[j]) * A*xi[i]
    c(i, j) = (B/2 + A*eta[j])^2
    e(i, j) = 2*A*H*b(i,j) / J(i, j)

    umap = reshape(1:N, n+1, n+1)     # Index mapping from 2D grid to vector
    Q = Tuple{Int64,Int64,Float64}[]  # Array of matrix elements (row,col,value)
    rhs = zeros(N)

    # Main loop, insert stencil in matrix for each node point
    for j = 1:n+1
        for i = 1:n+1
            row = umap[i,j]
            if i == n+1 || j == 1 
                # Dirichlet boundary condition, u = 0
                push!(Q, (row, row, 1.0))
                rhs[row] = 0.0
            elseif i == 1
                # Neumann, left boundary
                push!(Q, (row, row, -1.5))
                push!(Q, (row, umap[2, j], 2.0))
                push!(Q, (row, umap[3, j], -0.5))
                rhs[row] = 0.0
            elseif j == n+1
                # Neumann, top boundary
                push!(Q, (row, row, 1.5 * (B/2 + A)))
                push!(Q, (row, umap[i, n], -2.0 * (B/2 + A)))
                push!(Q, (row, umap[i, n-1], 0.5 * (B/2 + A)))
                push!(Q, (row, umap[i+1, n+1], -A*xi[i]/2))
                push!(Q, (row, umap[i-1, n+1], A*xi[i]/2))
                rhs[row] = 0.0
            else
                # Interior nodes, 9-point stencil
                push!(Q, (row, row, -2.0 * a(i, j) - 2.0 * c(i,j)))
                push!(Q, (row, umap[i-1,j], a(i,j) - h*e(i,j)/2))
                push!(Q, (row, umap[i+1,j], a(i,j) + h*e(i,j)/2))
                push!(Q, (row, umap[i,j+1], c(i,j)))
                push!(Q, (row, umap[i,j-1], c(i,j)))
                push!(Q, (row, umap[i-1,j-1], -b(i,j)/2))
                push!(Q, (row, umap[i-1,j+1], b(i,j)/2))
                push!(Q, (row, umap[i+1,j-1], b(i,j)/2))
                push!(Q, (row, umap[i+1,j+1], -b(i,j)/2))
 
                rhs[row] = -(J(i,j) * h)^2
            end
        end
    end

    # Create CSC sparse matrix from matrix elements
    Q = sparse((x->x[1]).(Q), (x->x[2]).(Q), (x->x[3]).(Q), N, N)

    u = Q \ rhs
    
    # Convert back to x, y
    tox(xi, eta) = (B/2 .+ A .* eta) .* xi
    toy(xi, eta) = H * eta

    xis = repeat(xi, n+1)
    etas = repeat(eta, inner=n+1)

    x = tox(xis, etas)
    y = toy(xis, etas)

    return Q, x, y, u
end

function problem2c()
    n = 20
    L = 3.0
    B = 0.5
    H = 1.0
    Q, x, y, u = channelFlow(L, B, H, n)
    fig, ax = subplots()
    ax.tricontour(x, y, u, levels=10, colors="k")
    pcm = ax.tricontourf(x, y, u, levels=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    axis("equal")
    fig.colorbar(pcm, ax=ax)
    display(fig)
end
#problem2c()


function findIndices(i, j, smalln, bign)
    return trunc(Int64,bign*(i-1)/smalln)+1, trunc(Int64,bign*(j-1)/smalln)+1
end    

function problem2d()
    fig, axs = subplots(2,3, figsize=(15,6))
    fig.suptitle("Convergence plots for skewed domain")
    L = 3.0
    H = 1.0
    ns = [10, 20, 40, 80, 160]
    bign = 320
    Bs = [0.0, 0.5, 1.0]
    for j = (1:3)
        Q, bigx, bigy, uexact = channelFlow(L, Bs[j], H, bign)
        uexact = reshape(uexact, bign+1, bign+1)
        errors = zeros(length(ns))
        hs = 1 ./ ns
    
        for i = (1:length(ns))
            n = ns[i]
            Q, x, y, u = channelFlow(L, Bs[j], H, n)
            u = reshape(u, n+1, n+1)
            
            difference = zeros(n+1,n+1)
            for row = (1:n+1)
                for col = (1:n+1)
                    bigi, bigj = findIndices(row, col, n, bign)
                    difference[row, col] = abs(u[row, col] - uexact[bigi, bigj])
                end
            end
            errors[i] = maximum(difference)
            
            # Plot difference for n = 160
            if i == length(ns)
                axs[2,j].set_title("Error")
                difference = reshape(difference, (n+1)^2)
                axs[2,j].tricontour(x, y, difference)
                pcm = axs[2,j].tricontourf(x, y, difference)
                axs[2,j].set_xlabel("x")
                axs[2,j].set_ylabel("y")
                fig.colorbar(pcm, ax=axs[2,j])
            end
        end
        fit = Polynomials.fit(log.(hs), log.(errors), 1)
        order = coeffs(fit)[2]
    
        axs[1,j].set_title("B=$(Bs[j]) , Convergence of order $(round(order,digits=3))")
        axs[1,j].loglog(hs, errors)
        axs[1,j].set_xlabel("h")
        axs[1,j].set_ylabel("error")
    end
    display(fig)
end

#problem2d()