module ReiterJulia

using Interpolations
using FastGaussQuadrature
using Distributions
using Roots
using QuadGK
using LinearAlgebra
using SparseArrays

export DiscretizedModel, Aggregates, Prices, Parameters, initParameters!, initDiscretizedModel!, decisionRule!, setTransition!, setPrices!, capitalSupply, excessCapital!, H, solve_eig, @load, @save

"""
	@load d v1 ...

Loads variables `v1`, ... from the structure `d`
"""
macro load(d, vars...)
	expr = Expr(:block)
	for v in vars
		push!(expr.args, :($v = $d.$v))
	end
	return esc(expr)
end

"""
	@save d v1 ...

Saves local variables `v1`, ... in the associated slot `d.v1` of structure `d`
"""
macro save(d, vars...)
	expr = Expr(:block)
	for v in vars
		push!(expr.args, :($d.$v = $v))
	end
	return esc(expr)
end

"""
    Aggregates

A structure for aggregate endogenous and exogenous variables
"""
mutable struct Aggregates
    K   # Aggregate capital
    z   # Aggregate TFP shock
    Aggregates() = new()
end

"""
    Parameters()

A structure for model parameters
"""
mutable struct Parameters
    α       # Share of capital in output
    β       # Discount factor
    δ       # Depreciation rate
    A       # Output scale
    N       # Aggregate labor
    ρ_z     # Persistence rate of the aggregate TFP shock
    σ_z     # Standard deviation of the aggregate TFP shock innovations
    μ       # Mean of the Normal law underpinning idiosyncratic productivity shocks
    σ       # Standard deviation of the Normal law underpinning idiosyncratic productivity shocks
    Parameters() = new()
end

"""
    initParameters!(params::Parameters)

Fills the `param` structure with canonical parameter values
"""
function initParameters!(params::Parameters)
    β = 0.95                # Discount factor   
    α = 0.33                # Share of capital in output
    δ = 0.1                 # Depreciation rate
    A = (1/β - 1 + δ)/α     # Output scale
    ρ_z = 0.8               # Persistence rate of the aggregate TFP shock
    σ_z = 0.014             # Standard deviation of the aggregate TFP shock innovations
    σ = 0.2                 # Standard deviation of the Normal law underpinning idiosyncratic productivity shocks
    μ = -0.5*σ^2            # Mean of the Normal law underpinning idiosyncratic productivity shocks
    N = exp(μ + 0.5*σ^2)    # Aggregate labor
    @save params β α δ A ρ_z σ_z σ μ N
end

"""
    Prices

A structure to store prices, e.g. wages, interest rates
"""
mutable struct Prices
    w   # wage
    r   # net interest rate
    R   # gross interest rate
    Prices() = new()
end

"""
    DiscretizedModel

A structure to store the parameters associated with the discretized model, e.g. asset and earning grids, Gauss-Hermite nodes and weights 
"""
mutable struct DiscretizedModel
    n_p     # Number of points of the grid for households' earnings
    χ       # Lowest earnings: binding liquidity constraint
    x_max   # maximum value for earnings 
    x       # grid for households' earnings
    x_fine  # grid for households' earnings - dense version for the approximation of the law of motion of the capital distribution
    n_d     # number of points of the histogram: discretized capital distribution
    k       # histogram: capital nodes                   
    k_min   # minimum value of the capital grid
    k_max   # maximum value of the capital grid
    Π       # histogram: Markov transition matrix
    n_ξ     # number of quadrature nodes
    ω       # quadrature weights for the idiosyncratic shock distribution
    ξ       # quadrature nodes for the idiosyncratic shock distribution
    DiscretizedModel() = new()
end

"""
    setPrices!(prices::Prices, agg::Aggregates, param::Parameters)

Fills the `prices` structure with the values derived from the aggregate variables and the parameters stored in `agg` and `param`
"""
function setPrices!(prices::Prices, agg::Aggregates, param::Parameters)
    @load param α δ A N
    @load agg z K
    r = A*exp(z)*α*(K/N)^(α-1) - δ
    w = A*exp(z)*(1-α)*(K/N)^α
    R = 1+r
    @save prices R r w
end

"""
    initDiscretizedModel!(d::DiscretizedModel, n_p, n_d, n_ξ, params)

Initializes the `DiscretizedModel` `d` structure with the canonical decision rule grid, histogram grid and Gauss-Hermite quadrature weights and nodes where
   - `n_p` is the number of nodes for the households' decision rule
   - `n_d` is the number of nodes for the capital histogram
   - `n_ξ` is the number of Gauss-Hermite nodes
"""
function initDiscretizedModel!(d::DiscretizedModel, n_p, n_d, n_ξ, params::Parameters)
    @save d n_p n_d n_ξ 
    # Quadrature nodes and weights for idiosyncratic shocks
    @load params μ σ
    ξ, ω = gaussHermite(μ, σ, n_ξ)
    @save d ξ ω

    # Nodes for the capital grid
    k_min = 0.
    k_max = 10.
    k = exp.(range(log(k_min+1), log(k_max+1), length=n_d+1)) .- 1.
    @save d k k_min k_max

    # Households' earnings grid 
    χ = 0.
    x_max = 0.5*k_max
    x = exp.(range(log(χ+1.), log(x_max+1.), length=n_p+1)) .- 1.
    @save d χ x x_max

end

"""
    ud(c; γ=1. )

The derivative of a CRRA utility function with parameter gamma taken at `c` 
"""
function ud(c; γ = 1.)
    return c^(-γ)
end

"""
    ud⁻¹(c; γ=1. )

The inverse function of the derivative of a CRRA utility function with parameter gamma taken at `c` 
"""
function ud⁻¹(c; γ = 1.)
    return c^(-1/γ)
end

"""
    gaussHermite(μ, σ, n)

Returns the nodes and weights of a Gauss-Hermite quadrature for a log-Normal law with parameters `μ` and `σ` and number of nodes `n`
"""
function gaussHermite(μ, σ, n)
    ξ, ω = gausshermite(n)
    ξ .= exp.(μ .+ sqrt(2)*σ.*ξ)
    return (ξ, ω./sqrt(π))
end

"""
    expEuler(sₜ₊₁, d::DiscretizedModel, Pₜ₊₁::Prices, params::Parameters)

Returns the expected term β𝔼ₜ[u'(cₜ₊₁)] with
   - `sₜ₊₁` is the households' decision rule taken at knot points
   - `d` is the considered `DiscretizedModel` instance
   - `Pₜ₊₁` is the `Prices` structure relevant on time `t+1`
   - `params` is the relevant `Parameters` instance  
"""
function expEuler(sₜ₊₁, d::DiscretizedModel, Pₜ₊₁::Prices, params::Parameters)
    @load d ξ ω x n_ξ n_p 
    @load Pₜ₊₁ R w
    @load params β

    # Computes all possible values of next period's earnings R*k + w*ξ
    earnings = repeat(sₜ₊₁, 1, n_ξ)
    @. earnings = R*earnings
    y = repeat(ξ', n_p+1, 1)
    @. earnings += w*y
    # Interpolates all the possible values of R*k + w*ξ
    k_hat = interpolate((x,), sₜ₊₁, Gridded(Linear()))
    k_hat = extrapolate(k_hat, Linear())
    # Impose non-negative consumption and no borrowing
    kₜ₊₁ = @. max(0., min( k_hat(earnings), earnings ))
    # Computes the value of c_next 
    cₜ₊₁ = earnings - kₜ₊₁ 
    return (β*R*(ud.(cₜ₊₁)*ω), k_hat)
end

"""
    decisionRule!(s, d::DiscretizedModel, P::Prices, params::Parameters; verbose=false)

Fills `s` with the steady-state decision rule vector where
   - `d` is the considered `DiscretizedModel` instance
   - `P` is the `Prices` structure for the steady state
   - `params` is the relevant `Parameters` instance  
"""
function decisionRule!(s, d::DiscretizedModel, P::Prices, params::Parameters; verbose=false)
    @load d x x_max χ n_p
    ϵ = eps()
    dist = 1e6
    iter = 0
    max_iter = 2000
    while (dist > ϵ) && (iter < max_iter+1)
        if verbose && ((iter % 50 == 0) || (iter == max_iter)) 
            println("i = ", iter, ": ", dist)
        end
        # Deriving the Euler equation error
        # Note that current period's capital is s
        Eudₜ₊₁, k_hat = expEuler(s, d, P, params)
        s_new = x .- ud⁻¹.(Eudₜ₊₁)
        @. s_new = max(0., min(x, s_new))
        dist = maximum(abs.(s_new-s))
        @. s = s_new
        iter += 1
        ind_χ = findfirst(s .> 0.)
        if ind_χ > 1
            χ = x[ind_χ-1]
            x .= exp.(range(log(χ+1.), log(x_max+1.), length=n_p+1)) .- 1.
            @. s = k_hat(x)
        end
    end
    if iter == max_iter
        println("Convergence failed!")
    end
    println("i = ", iter, ": ", dist)
    @save d x χ
end

"""
    setTransition!(d::DiscretizedModel, prices::Prices, params::Parameters, s)

Returns the Markov transition matrix and fills the `x_fine` field in the `DiscretizedModel` `d` structure
   - `prices` is the considered steady-state `Prices` structure
   - `params` is the considered `Parameters` structure
"""
function setTransition!(d::DiscretizedModel, prices::Prices, params::Parameters, s)
    @load d n_d k x χ k_min
    @load prices w R 
    @load params μ σ
    # %%
    F(x) = cdf(LogNormal(μ, σ),x)
    Fʸ(x) = F(x/w)
    F¹(a,b) = quadgk(F, a, b)[1]
    # %%
    k_hat = interpolate((x,), s, Gridded(Linear()))
    k_hat = extrapolate(k_hat, Linear())
    x_fine = zeros(n_d+1)
    x_fine[1] = χ
    for i=2:n_d+1
        x_fine[i] = find_zero(x->k_hat(x) - k[i], (0., 2*k[end]))
    end
    @save d x_fine
    # %%
    row = Int64[]
    col = Int64[]
    V = Float64[]
    push!(row, 1)
    push!(col, 1)
    push!(V, Fʸ(χ-R*k_min))
    # %%
    j = 2
    ξ2 = (χ - R*k[j-1])/w
    ξ1 = (χ - R*k[j])/w
    # %%
    while (ξ1 > 0.) && (j <= n_d+1)
        push!(row, j)
        push!(col, 1)
        push!(V, F¹(ξ1,ξ2)*w/(R*(k[j]-k[j-1])))
        j += 1
        ξ2 = (χ - R*k[j-1])/w
        ξ1 = (χ - R*k[j])/w
    end
    # %%
    for i = 2:n_d+1
        v = Fʸ(x_fine[i] - R*k_min) - Fʸ(x_fine[i-1] - R*k_min)
        if v > eps()
            push!(row, 1)
            push!(col, i)
            push!(V, v)
        end
    end
    # %%
    for i = 2:n_d+1
        for j = 2:n_d+1
            ξ_i_j = (x_fine[i] - R*k[j])/w
            ξ_im1_j = (x_fine[i-1] - R*k[j])/w
            ξ_i_jm1 = (x_fine[i] - R*k[j-1])/w
            ξ_im1_jm1 = (x_fine[i-1] - R*k[j-1])/w
            if max(ξ_i_j, ξ_im1_j, ξ_i_jm1, ξ_im1_jm1) > 0.
                v = (F¹(ξ_i_j, ξ_i_jm1) - F¹(ξ_im1_j, ξ_im1_jm1))*w/(R*(k[j]-k[j-1])) 
                if v > eps()
                    push!(row,j)
                    push!(col,i)
                    push!(V,v)
                end
            end
        end
    end
    Π = sparse(row, col, V, n_d+1, n_d+1)
    return Π
end

"""
    capitalSupply(p, d::DiscretizedModel)

Returns the value of capital supply stemming from the histogram `p` and the capital grid `d.k`
"""
function capitalSupply(p, d::DiscretizedModel)
    @load d k
    k_min = k[begin]
    K = p[1]*k_min
    for i in 2:length(p)
        K += 0.5*p[i]*(k[i]+k[i-1])
    end
    return K
end

"""
    excessCapital!(d::DiscretizedModel, agg::Aggregates, prices::Prices, params::Parameters, s, K; verbose=false)

Returns the difference between `K` and the supply of capital `K` induces:
   1. store the `K` value in the `Aggregates` `agg` structure
   2. fill the `prices` structure using the `setPrices!` function
   3. compute the households' decision rule using `decisionRule!`
   4. get the transition function `setTransition!` and the associated steady-state histogram
   5. compute the resulting capital supply using the `capitalSupply` routine
"""
function excessCapital!(d::DiscretizedModel, agg::Aggregates, prices::Prices, params::Parameters, s, K; verbose=false)
    agg.K = K
    setPrices!(prices, agg, params)
    decisionRule!(s, d, prices, params, verbose=verbose)
    Π = setTransition!(d, prices, params, s)
    @load d n_d
    A = Π - I(n_d+1)
    A[:,end] .= 1
    b = zeros(n_d+1)
    b[end] = 1
    p = A'\b
    agg.K = capitalSupply(p, d)
    return agg.K - K 
end

"""
    H(xₜ, xₜ₊₁, yₜ, yₜ₊₁, d::DiscretizedModel, params::Parameters)

Returns the system pinpointing the equilibrium of the economy:
   - x collects pre-determined and exogenous variables
   - y collects control variables
   - `d` is the relevant `DiscretizedModel` instance
   - `params` is the relevant `Parameters` instance
"""
function H(xₜ, xₜ₊₁, yₜ, yₜ₊₁, d::DiscretizedModel, params::Parameters)
    @load d n_p n_d x χ 
    @load params ρ_z σ_z
    resid = zeros(n_p+n_d+3)
    # %%
    # x collects pre-determined and exogenous variables (state variables). It includes the histogram p, the aggregate capital and the aggregate shock z
    # y collects non pre-determined variables (control variables)
    # %%
    pₜ = vcat(1-sum(xₜ[1:n_d]), xₜ[1:n_d] ) # x contains all the elements of the histogram, except one (for linear dependence reasons)
    Kₜ = xₜ[n_d+1]
    zₜ = xₜ[n_d+2]
    # %%
    pₜ₊₁ = vcat(1-sum(xₜ₊₁[1:n_d]), xₜ₊₁[1:n_d])
    Kₜ₊₁ = xₜ₊₁[n_d+1]
    zₜ₊₁ = xₜ₊₁[n_d+2]
    # %%
    sₜ = yₜ
    sₜ₊₁ = yₜ₊₁
    # %%
    Aₜ₊₁ = Aggregates()
    Aₜ₊₁.K = Kₜ₊₁
    Aₜ₊₁.z = zₜ₊₁
    # %%
    Aₜ = Aggregates()
    Aₜ.K = Kₜ
    Aₜ.z = zₜ
    # %%
    Pₜ₊₁ = Prices()
    setPrices!(Pₜ₊₁, Aₜ₊₁, params)
    # %%
    Pₜ = Prices()
    setPrices!(Pₜ, Aₜ, params)
    # %%
    # Euler equation residuals
    Eudₜ₊₁, _ = expEuler(sₜ₊₁, d, Pₜ₊₁, params)
    @. resid[1:n_p+1] = sₜ -  max(0., min(x, x - ud⁻¹(Eudₜ₊₁)))
    # %%
    # Distribution dynamics
    Πₜ = setTransition!(d, Pₜ, params, sₜ)
    resid[n_p+2:n_p+n_d+1] .= reshape(pₜ₊₁' - pₜ'*Πₜ, n_d+1)[2:end]
    # %%
    # Aggregate capital K
    K_supply = capitalSupply(pₜ₊₁, d)
    resid[n_p+n_d+2] = Kₜ₊₁ - K_supply 
    # %%
    # Aggregate shock process
    resid[n_p+n_d+3] = zₜ₊₁ - ρ_z*zₜ
    # %%
    return resid
end

"""
    solve_eig(A::Array{T,2}, B::Array{T,2}, n_x::Int) where T<: AbstractFloat

Returns the transition and policy functions following the method SGU (2004) delineates. `A` and `B` are defined as in SGU. `n_x` is the number of states.
"""
function solve_eig(A::Array{T,2}, B::Array{T,2}, n_x::Int) where T<: AbstractFloat

    F = eigen(B,A)
    perm = sortperm(abs.(F.values))
    V = F.vectors[:,perm]
    D = F.values[perm]
    m = findlast(abs.(D) .< 1)
    eu = [true,true]

    if m > n_x
        eu[2] = false
        println("WARNING: the equilibrium is not unique !")
    elseif m < n_x
        eu[1] = false
        println("WARNING: the equilibrium does not exist !")
    end

    if all(eu)
        h_x = V[1:m,1:m]*Diagonal(D)[1:m,1:m]*inv(V[1:m,1:m])
        g_x = V[m+1:end,1:m]*inv(V[1:m,1:m])
        return (real.(g_x), real.(h_x), eu)
    end

end

end 
