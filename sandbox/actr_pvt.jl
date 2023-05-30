##############################################################################################################
#                                           load packages
##############################################################################################################
cd(@__DIR__)
using Pkg
Pkg.activate("..")
using Revise
using NormalizingFlowsTutorials
using InvertibleNetworks: NetworkConditionalGlow
using Distributions
using ACTRPVT
using Random
using PyPlot
Random.seed!(02102)
##############################################################################################################
#                                           generate data
##############################################################################################################
n_obs = 100
parms = (υ=4.0, τ=3.0, λ=.98, γ=.04)
_,rts = rand(PVTModel(;parms...), n_obs)
##############################################################################################################
#                                           generate training data
##############################################################################################################
function sample_prior()
    υ = rand(truncated(Normal(4.0, 1.5), 0, Inf))
    τ = rand(truncated(Normal(3.0, 1.5), 0, Inf))
    λ = rand(Beta(49, 1))
    γ = rand(truncated(Normal(.04, .02), 0, Inf))
    return [υ,τ,λ,γ]
end

function sample(parms, n_obs)
    _,rts = rand(PVTModel(parms...), n_obs)
    return rts 
end

n_parms = 4 
n_train = 20_000
x_train = mapreduce(x -> sample_prior(), hcat, 1:n_train)
y_train = mapreduce(i -> sample(x_train[:,i], n_obs), hcat, 1:n_train)
##############################################################################################################
#                                          sample prior distribution
##############################################################################################################
n_samples = 1000
x_prior = mapreduce(x -> sample_prior(), hcat, 1:n_samples)'
##############################################################################################################
#                                           train neural network
##############################################################################################################
n_epochs = 30
batch_size = 1000
n_batches = div(n_train, batch_size)
n_hidden = 32
n_multiscale = 3
n_coupling = 4
network = NetworkConditionalGlow(n_parms, n_obs, n_hidden, n_multiscale, n_coupling)
losses = train!(network, x_train, y_train; n_epochs, n_batches, batch_size, n_obs)

fig = figure()
plot(losses)
xlabel("iterations")
ylabel("loss")
fig
##############################################################################################################
#                                sample from posterior distribution
##############################################################################################################
x_post = sample_posterior(network, rts; n_parms, n_samples)
##############################################################################################################
#                                        plot results
##############################################################################################################
fig = figure()
subplot(2,2,1)
hist(x_prior[:,1];alpha=0.7,density=true,label="Prior")
hist(x_post[:,1];alpha=0.7,density=true,label="Posterior")
axvline(parms.υ, color="k", linewidth=1,label="Ground truth")
xlabel(L"\upsilon"); ylabel("Density"); 
legend()

subplot(2,2,2)
hist(x_prior[:,2]; alpha=0.7,density=true,label="Prior")
hist(x_post[:,2]; alpha=0.7,density=true,label="Posterior")
axvline(parms.τ, color="k", linewidth=1,label="Ground truth")
xlabel(L"\tau"); ylabel("Density");
legend()

subplot(2,2,3)
hist(x_prior[:,3]; alpha=0.7,density=true,label="Prior")
hist(x_post[:,3]; alpha=0.7,density=true,label="Posterior")
axvline(parms.λ, color="k", linewidth=1,label="Ground truth")
xlabel(L"\lambda"); ylabel("Density");
legend()

subplot(2,2,4)
hist(x_prior[:,4]; alpha=0.7,density=true,label="Prior")
hist(x_post[:,4]; alpha=0.7,density=true,label="Posterior")
axvline(parms.γ, color="k", linewidth=1,label="Ground truth")
xlabel(L"\gamma"); ylabel("Density");
legend()

tight_layout()
fig


n_obs = 100
parms = (υ=5.0, τ=3.0, λ=.98, γ=.04)
_,rts = rand(PVTModel(;parms...), n_obs)
x_post = sample_posterior(network, rts; n_parms, n_samples)
println(round.(mean(x_post, dims=1), digits=3))
parms