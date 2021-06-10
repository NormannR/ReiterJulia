using Roots
using LinearAlgebra
using SparseArrays
import FiniteDiff.finite_difference_jacobian
using Plots
pyplot()
# %%
using ReiterJulia
# %%
n_p = 100
n_d = 200
n_ξ = 41
# %%
d = DiscretizedModel()
params = Parameters()
initParameters!(params)
P = Prices()
A = Aggregates()
z = 0.
@save A z
# %%
initDiscretizedModel!(d, n_p, n_d, n_ξ, params)
# %%
s = 0.8*d.x  # houdeholds' policy function for capital taken at points in the earnings grid x
# %%
Kss = find_zero(x->excessCapital!(d, A, P, params, s, x, verbose=false), (0.95, 1.05), verbose=true, xatol = eps())
# %%
A.K = Kss
# %%
Π = setTransition!(d, P, params, s)
M = Π - I(d.n_d+1)
M[:,end] .= 1
b = zeros(d.n_d+1)
b[end] = 1
p = M'\b
# %%
println("Check steady-state residual: ", A.K - capitalSupply(p,d))
# %%
plot(d.x_fine, p, color="black", title = "Steady-state capital distribution")
xgrid!(:off)
ygrid!(:off)
# %%
x = vcat(p[2:end], [A.K, A.z])
y = s
# %%
H_x = finite_difference_jacobian(u->H(u,x,y,y,d,params), x)
# %%
H_xp = finite_difference_jacobian(u->H(x,u,y,y,d,params), x)
# %%
H_y = finite_difference_jacobian(u->H(x,x,u,y,d,params), y)
# %%
H_yp = finite_difference_jacobian(u->H(x,x,y,u,d,params), y)
# %%
AA = hcat(H_xp, H_yp)
BB = -hcat(H_x, H_y)
# %%
# g_x is the jacobian of the decision function (controls) wrt states
# h_x is the jacobian of the transition function (states) wrt states 
g_x, h_x, eu = solve_eig(AA, BB, d.n_d+2)
# %%
T = 60
dx = zeros(n_d+2,T)
dx[end,1] = 0.1
# %%
for t in 2:T
    dx[:,t] .= h_x*dx[:,t-1]
end
# %%
plot(d.k[2:end], dx[1:n_d,1]+x[1:n_d], label="t=0")
plot!(d.k[2:end], dx[1:n_d,20]+x[1:n_d], label="t=20")
plot!(d.k[2:end], dx[1:n_d,40]+x[1:n_d], label="t=40")
xgrid!(:off)
ygrid!(:off)
xlabel!("Capital distribution")
# %%
plot(1:T, x[end] .+ dx[end-1,:])
xlabel!("Time")
ylabel!("Aggregate capital")
xgrid!(:off)
ygrid!(:off)
# %%