function train!(network::NetworkConditionalGlow, x_train, y_train; 
        n_batches = 200,
        batch_size = 1000,
        n_epochs = 2,
        n_train,
        show_progress = true)

    progress = Progress(n_batches * n_epochs)
    opt = ADAM(4f-3)
    # Training logs 
    loss_l2   = Float64[]; logdet_train = Float64[];
    idx_e = reshape(1:n_train, batch_size, n_batches) 
    for e ∈ 1:n_epochs 
        for b ∈ 1:n_batches 
            X = x_train[:, :, :, idx_e[:,b]]
            Y = y_train[:, :, :, idx_e[:,b]]
        
            # Forward pass of normalizing flow
            Zx, Zy, lgdet = network.forward(X, Y)

            # Loss function is l2 norm - logdet
            append!(loss_l2, norm(Zx)^2 / prod(size(X)))  # normalize by image size and batch size
            append!(logdet_train, -lgdet / prod(size(X)[1:end-1])) # logdet is already normalized by batch size

            # Set gradients of flow
            network.backward(Zx / batch_size, Zx, Zy)

            # Update parameters of flow
            for p in get_params(network) 
                Flux.update!(opt,p.data,p.grad)
            end 
            clear_grad!(network)

            showvalues = [
                (:epoch, "$e/$n_epochs"),
                (:batch, "$b/$n_batches"),
                (:f_l2, round(loss_l2[end],digits=3)),
                (:lgdet, round(logdet_train[end],digits=3)),
                (:full_objective, round(loss_l2[end] + logdet_train[end],digits=3))]

            show_progress ? next!(progress; showvalues) : nothing
        end
    end
    return network
end

function sample_posterior(G, data; n_parms, n_samples)
    ZX_noise = randn(1, 1, n_parms, n_samples) 
    Y_forward = repeat(data, 1, 1, 1, n_samples) 
    _, Zy_fixed_train, _ = G.forward(ZX_noise, Y_forward); #needed to set the proper transforms on inverse
    post_samples =  G.inverse(ZX_noise, Zy_fixed_train)
    return reshape(post_samples, (n_parms,n_samples)) |> transpose
end