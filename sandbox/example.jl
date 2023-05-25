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
using Random
using PyPlot
Random.seed!(8744)
##############################################################################################################
#                                           generate training data
##############################################################################################################
function sample_prior()
    μ = rand(Normal(0, 1))
    σ = rand(Uniform(0, 10))
    return [μ,σ]
end

n_parms = 2 
n_obs = 50
n_train = 10000
# train using samples from joint distribution x,y ~ p(x,y) where x=[μ, σ] -> y = N(μ, σ)
# rows: μ, σ, y
x_train = mapreduce(x -> sample_prior(), hcat, 1:n_train)
y_train = mapreduce(i -> rand(Normal(x_train[:,i]...), n_obs), hcat, 1:n_train)
x_train = reshape(x_train, (1,1,n_parms,:))
y_train = reshape(y_train, (1,1,n_obs,:));
##############################################################################################################
#                                           train neural network
##############################################################################################################
n_epochs = 2
batch_size = 200
n_batches = div(n_train, batch_size)
n_hidden = 32
n_multiscale = 3
n_coupling = 4
network = NetworkConditionalGlow(n_parms, n_obs, n_hidden, n_multiscale, n_coupling)
train!(network, x_train, y_train;
     n_epochs, n_batches, batch_size, n_train)
##############################################################################################################
#                                sample from posterior distribution
##############################################################################################################
# posterior inference on unseen observation
x_ground_truth = [-1,2] # mu=-1, sigma=2
n_samples = 1000
data = reshape(rand(Normal(x_ground_truth[1], x_ground_truth[2]), n_obs), (1,1,n_obs,:))

x_post = sample_posterior(network, data; n_parms, n_samples)
##############################################################################################################
#                                        plot results
##############################################################################################################
x_prior = mapreduce(x -> sample_prior(), hcat, 1:n_samples)'

fig = figure()
subplot(1,2,1)
hist(x_prior[:,1];alpha=0.7,density=true,label="Prior")
hist(x_post[:,1];alpha=0.7,density=true,label="Posterior")
axvline(x_ground_truth[1], color="k", linewidth=1,label="Ground truth")
xlabel(L"\mu"); ylabel("Density"); 
legend()

subplot(1,2,2)
hist(x_prior[:,2]; alpha=0.7,density=true,label="Prior")
hist(x_post[:,2]; alpha=0.7,density=true,label="Posterior")
axvline(x_ground_truth[2], color="k", linewidth=1,label="Ground truth")
xlabel(L"\sigma"); ylabel("Density");
legend()
tight_layout()
fig