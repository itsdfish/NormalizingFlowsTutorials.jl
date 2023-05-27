function train!(network::NetworkConditionalGlow, 
    x_train::AbstractArray{T,2}, y_train::AbstractArray{T,2}; n_obs, kwargs...) where {T}
    x_train = reshape(x_train, (1,1,size(x_train, 1),:))
    y_train = reshape(y_train, (1,1,n_obs,:))
    return train!(network, x_train, y_train; kwargs...)
end

function train!(network::NetworkConditionalGlow, x_train, y_train; 
        n_batches = 200,
        batch_size = 1000,
        n_epochs = 2,
        show_progress = true)

    n_iter = n_batches * n_epochs
    iter = 1
    progress = Progress(n_iter)
    opt = ADAM(4f-3)
    # Training logs 
    loss_l2 = fill(0.0, n_iter)
    logdet_train = fill(0.0, n_iter)
    for e ∈ 1:n_epochs 
        for b ∈ 1:n_batches 
            idx_e = (batch_size * (b - 1) + 1):(batch_size * b)
            X = x_train[:, :, :, idx_e]
            Y = y_train[:, :, :, idx_e]

            # Forward pass of normalizing flow
            Zx, Zy, lgdet = network.forward(X, Y)

            # Loss function is l2 norm - logdet
            loss_l2[iter] = norm(Zx)^2 / prod(size(X))  # normalize by image size and batch size
            logdet_train[iter] = -lgdet / prod(size(X)[1:end-1]) # logdet is already normalized by batch size

            # Set gradients of flow
            network.backward(Zx / batch_size, Zx, Zy)

            # Update parameters of flow
            for p in get_params(network) 
                Flux.update!(opt, p.data, p.grad)
            end 
            clear_grad!(network)

            showvalues = [
                (:epoch, "$e/$n_epochs"),
                (:batch, "$b/$n_batches"),
                (:f_l2, round(loss_l2[iter],digits=3)),
                (:lgdet, round(logdet_train[iter],digits=3)),
                (:loss, round(loss_l2[iter] + logdet_train[iter],digits=3))]

            show_progress ? next!(progress; showvalues) : nothing
            iter += 1
        end
    end
    return loss_l2 + logdet_train
end

function sample_posterior(network, data::AbstractArray{T,1}; kwargs...) where {T}
    data = reshape(data, (1,1,length(data),:))
    return sample_posterior(network, data; kwargs...)
end

function sample_posterior(network, data; n_parms, n_samples)
    ZX_noise = randn(1, 1, n_parms, n_samples) 
    Y_forward = repeat(data, 1, 1, 1, n_samples) 
    _, Zy_fixed_train, _ = network.forward(ZX_noise, Y_forward) #needed to set the proper transforms on inverse
    post_samples = network.inverse(ZX_noise, Zy_fixed_train)
    return reshape(post_samples, (n_parms,n_samples)) |> transpose
end