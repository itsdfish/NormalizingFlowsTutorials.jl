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
using LinearAlgebra
using Random
using PyPlot
Random.seed!(8744)
##############################################################################################################
#                                           generate data
##############################################################################################################
# posterior inference on unseen observation
μ = [-1,1]
n_dim = length(μ)
σ = 1
n_obs = 50

# what shape should this be?
data = reshape(rand(MvNormal(μ, σ * I(2)), n_obs), (n_obs,n_dim,1))
##############################################################################################################
#                                           generate training data
##############################################################################################################
function sample_prior()
    μ = rand(Normal(0, 1), 2)
    σ = rand(truncated(LogNormal(1, 1), 0, 10))
    return [μ...,σ]
end

n_samples = 1000
n_parms = 4 
n_train = 10_000
# train using samples from joint distribution x,y ~ p(x,y) where x=[μ, σ] -> y = N(μ, σ)
# rows: μ, σ, y
x_train = mapreduce(x -> sample_prior(), hcat, 1:n_train)

y_train = zeros(n_obs, n_dim, n_train)
for i ∈ 1:n_train
    y_train[:,:,i] = rand(MvNormal(x_train[1:2,i],x_train[3,i] * I(2)), n_obs)'
end
x_train = [x_train[1,:]'; x_train]
x_train = reshape(x_train, (1,4,n_train))
# hack to make even number of parameters. 
##############################################################################################################
#                                          sample prior distribution
##############################################################################################################
x_prior = mapreduce(x -> sample_prior(), hcat, 1:n_samples)'
##############################################################################################################
#                                           train neural network
##############################################################################################################
n_epochs = 10
batch_size = 1000
n_batches = div(n_train, batch_size)
n_hidden = 32
n_multiscale = 3
n_coupling = 4
network = NetworkConditionalGlow(n_parms, n_obs, n_hidden, n_multiscale, n_coupling)
losses = train!(network, x_train, y_train; n_epochs, n_batches, batch_size)
fig = figure()
plot(losses)
xlabel("iterations")
ylabel("loss")
fig
##############################################################################################################
#                                sample from posterior distribution
##############################################################################################################
x_post = sample_posterior(network, data; n_parms, n_samples)
##############################################################################################################
#                                        plot results
##############################################################################################################
fig = figure()
subplot(1,2,1)
hist(x_prior[:,1];alpha=0.7,density=true,label="Prior")
hist(x_post[:,1];alpha=0.7,density=true,label="Posterior")
axvline(μ[1], color="k", linewidth=1,label="Ground truth")
xlabel(L"\mu_1"); ylabel("Density"); 
legend()

fig = figure()
subplot(1,2,1)
hist(x_prior[:,2];alpha=0.7,density=true,label="Prior")
hist(x_post[:,2];alpha=0.7,density=true,label="Posterior")
axvline(μ[2], color="k", linewidth=1,label="Ground truth")
xlabel(L"\mu_2"); ylabel("Density"); 
legend()


subplot(1,2,3)
hist(x_prior[:,3]; alpha=0.7,density=true,label="Prior")
hist(x_post[:,3]; alpha=0.7,density=true,label="Posterior")
axvline(σ, color="k", linewidth=1,label="Ground truth")
xlabel(L"\sigma"); ylabel("Density");
legend()
tight_layout()
fig