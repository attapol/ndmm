# Experiment wrapper functions
#
# The main function is run_multinet to run an experiment based on
# the configuration in the config json. 
# This file is called from the experiment scripts in the exp_scripts folder
# The main function wraps around the training function and evaluating function,
# which is convenient for experimentation.
# 
# You can call it directly from Julia interpreter but 
# use exp_scripts/run_vsall_multideepnet.jl or something similar instead
#
# This file is part of Net module
#
function run_multinet(config_dict)
    return run_multinet(-1, config_dict)
end

function run_multinet(label_index, config_dict)
    # Run Experiment
    # If label_index is set to -1, then n-way classification
    
    # Model saving
    save_model = get_config_value(config_dict, "save_model", true)
    if save_model
        model_file_name = config_dict["file_name_prefix"] * "_label$label_index.model" 
        model_file = open(model_file_name, "w")
    else
        model_file_name = ""
    end

    # Logging
    log_to_file = get_config_value(config_dict, "log_to_file", true)
    if log_to_file
        log_file_name = config_dict["file_name_prefix"] * "_label$label_index.log" 
        log_stream = open(log_file_name, "w")
    else
        log_stream = STDOUT
    end

    # Constructing the net and setting up parameters
    learning_rate = get_config_value(config_dict, "learning_rate", 0.01)
    lr_smoother = get_config_value(config_dict, "lr_smoother", 0.00001)
    num_iterations = config_dict["num_iterations"]
    minibatch_size = get_config_value(config_dict, "minibatch_size", 0)
    pretraining_max_iterations = get_config_value(config_dict, "pretraining_max_iterations", 0)
    net, data, target, dev_data, dev_target, test_data, test_target = construct_net(label_index, config_dict)

    # Pretraining
    net.io = log_stream
    if pretraining_max_iterations > 0
        Net.feed_input_layer(net, data)
        Net.pretrain(net, pretraining_max_iterations)
    end

    # Reweighting and training
    tic()
    reweighting_factor = get_config_value(config_dict, "reweighting_factor", 0)
    target_weights = compute_inverse_ratio_weights(target, reweighting_factor)
    #target_weights = compute_balanced_weights(target, reweighting_factor)
    losses, dev_confusion_matrices, test_confusion_matrices = Net.train(net, data, 
        target, target_weights, dev_data, dev_target, test_data, test_target, 
        learning_rate, lr_smoother, num_iterations, minibatch_size)

    # Report and save
    total_seconds = toq()
    total_minutes = total_seconds / 60.0
    total_hours = total_minutes / 60.0
    spare_minutes = total_minutes % 60.0
    println(log_stream, "Total training time $total_seconds secs ($total_minutes minutes) ($total_hours hours and $spare_minutes minutes.)")
    net.io = []
    if save_model
        serialize(model_file, net)
    end
    return (losses, dev_confusion_matrices, test_confusion_matrices)
end

function get_data_target(label_index, layer_spec, prefix="")
    if prefix != ""
        prefix = prefix * "_"
    end

    println("Reading in data")
    if haskey(layer_spec, "$(prefix)data") 
        data, target = Net.read_sparse_matrix(layer_spec["$(prefix)data"])
    elseif haskey(layer_spec, "$(prefix)data_sparse")
        data, target = Net.read_sparse_matrix(layer_spec["$(prefix)data_sparse"])
    elseif haskey(layer_spec, "$(prefix)data_dense")
        data, target = Net.read_dense_matrix(layer_spec["$(prefix)data_dense"])
    else
        return None, None
    end
    println("Converting target matrix")
    target = _convert_target_matrix(label_index, target)
    return (data, target)
end

function construct_net(config_dict)
    return construct_net(-1, config_dict)
end

function construct_net(label_index, config_dict)
    # Construct a net based on the configuration
    # to support label_index vs all classification 
    #
    # if label_index is set to -1, then the net is set up for n-way classification
    #
    architecture = config_dict["architecture"]
    layers = Layer[]
    num_cases_list = Int64[]

    data_matrices = {} 
    target_matrices = Array[] 

    dev_data_matrices = {}
    dev_target_matrices = Array[]

    test_data_matrices = {}
    test_target_matrices = Array[]
    
    #initializing each individual layer
    for layer_spec in architecture
        if layer_spec["type"] == "input"
            println("Creating input layers")
            data, target = get_data_target(label_index, layer_spec, "")
            dev_data, dev_target = get_data_target(label_index, layer_spec, "dev")
            test_data, test_target = get_data_target(label_index, layer_spec, "test")

            push!(data_matrices, data)
            push!(target_matrices, target)

            push!(dev_data_matrices, dev_data)
            push!(dev_target_matrices, dev_target)

            push!(test_data_matrices, test_data)
            push!(test_target_matrices, test_target)

            push!(num_cases_list, size(data,1))
            push!(layers, Net.InputLayer(size(data, 2)))
        elseif layer_spec["type"] == "hidden" || layer_spec["type"] == "hidden_fixed"
            println("Creating hidden layers")
            activation_fn_name = get_config_value(config_dict, "activation_fn", "sigmoid")
            activation_fn, activation_prime_fn = get_activation_functions(activation_fn_name)
            hidden_layer = Net.HiddenLayer(layer_spec["num_units"])
            hidden_layer.activation_fn = activation_fn
            hidden_layer.activation_prime_fn = activation_prime_fn
            push!(layers, hidden_layer)
        elseif layer_spec["type"] == "output"
            #fix to just binary classifications for now
            println("Creating output layers")
            num_labels = size(target_matrices[1], 2)
            push!(layers, Net.OutputLayer(num_labels))
        elseif layer_spec["type"] == "gated_output"
            num_labels = size(target_matrices[1], 2)
            push!(layers, Net.GatedOutputLayer(num_labels))
        end
    end

    # perform checking on the training set
    _check_target_matrices(target_matrices)
    _check_target_matrices(dev_target_matrices)
    _check_target_matrices(test_target_matrices)

    # set up the network from the layers
    net = Net.MultiNet() 
    for i in 1:length(layers)
        layer_spec = architecture[i]
        layer = layers[i]
        # we assume a nice topological ordering in the architecture json
        # or some of these things will break especially parent connections
        if layer_spec["type"] == "input"
            Net.add_input_layer(net, layer)
            @assert length(layer_spec["parents"])==0 "Input layer cannot have parents"
        elseif layer_spec["type"] == "hidden" || layer_spec["type"] == "hidden_fixed"
            Net.add_layer(net, layer)
            for parent in layer_spec["parents"]
                Net.connect(net, parent, i)
            end
        elseif layer_spec["type"] == "output" 
            #we assume that gated output will override the previous setting of output layers
            Net.add_output_layer(net, layer)
            for parent in layer_spec["parents"]
                Net.connect(net, parent, i)
            end
        elseif layer_spec["type"] == "gated_output"
            Net.add_output_layer(net, layer)
            for expert_index in layer_spec["experts"]
                push!(layer.experts, layers[expert_index])
            end
            for parent in layer_spec["parents"]
                Net.connect(net, parent, i)
            end
        end
    end
    Net.initialize(net)
    return (net, data_matrices, target_matrices[1], dev_data_matrices, dev_target_matrices[1], test_data_matrices, test_target_matrices[1])
end

function _read_test_data(test_set_file_names, label_index)
    data_matrices = SparseMatrixCSC[]
    target_matrices = Array[]
    for test_set_file_name in test_set_file_names
        data, target = Net.read_sparse_matrix(test_set_file_name)
        target = [target[:, label_index] (target[:, label_index] + 1)%2]
        push!(data_matrices, data)
        push!(target_matrices, target)
    end
    return data_matrices, target_matrices[1]
end

function _convert_target_matrix(label_index, target)
    # Convert a target matrix
    #
    # one-vs-all classification when 0 <= label_index <= num labels - 1
    # For example, when label_index = 1
    # [1, 0, 0      --> [0 1
    #  0, 0, 1           0 1
    #  0, 1, 0           1 0
    #  0, 1, 0           1 0
    #  1, 0, 0 ]         0 1]
    #
    # The matrix is redundant but we afford to be redundant
    #
    println(size(target))
    if label_index != -1
        return [target[:, label_index + 1] (target[:, label_index + 1] + 1)%2]
    else
        return target
    end
end


function _check_target_matrices(target_matrices)
    #we want to check that they are all the same
    if length(target_matrices) > 1
        for i in 2:length(target_matrices)
            @assert target_matrices[i-1] == target_matrices[i] "All target should be the same across corresponding input layers"
        end
    end
end

function get_activation_functions(activation_function_name)
    if activation_function_name == "sigmoid"
        return (sigmoid, sigmoid_prime)
    elseif activation_function_name == "relu"
        return (relu, relu_prime)
    else
        error ("unknown activation function: $activation_function_name")
    end

end

# Use the config values from the config dictionary or use the specified default value
function get_config_value(config_dict, key, default_value)
    if !haskey(config_dict, key)
        println("$key not found. use default value of $default_value.")
        return default_value
    else
        return config_dict[key]
    end
end

function compute_inverse_ratio_weights(target, reweighting_factor)
    # Compute inverse proportion and use it as a weights
    # If class x has N_x instances and the dataset has k classes, each instance receives N * k / N_x 
    target_distribution = sum(target, 1)
    target_weights = ones(size(target, 1))
    num_rows = size(target, 1)
    num_classes = size(target, 2)
    for (i in 1:size(target_weights, 1))
        label_index = find(target[i,:])
        @assert  length(label_index) == 1
        label_index = label_index[1]
        inv_proportion = num_rows / target_distribution[label_index] / num_classes
        target_weights[i] = reweighting_factor * inv_proportion + (1 - reweighting_factor) * 1
    end
    return target_weights
end 


function compute_balanced_weights(target, ratio=1)
    # DEPRECATED
    # compute the weight for each instance such that the class distribution is balanced
    # The formula only makes sense for the binary cases only
    #
    # This function is used for producing the results in the first draft of NDMM paper
    target_distribution = sum(target, 1)
    minority_size = minimum(target_distribution)
    majority_size = maximum(target_distribution)
    target_weights = ones(size(target, 1))
    for (i in 1:size(target_weights,1))
        if target[i, 1] == 0
            current_target = 2
        else
            current_target = 1
        end
        if target_distribution[current_target] == majority_size
            target_weights[i] = ratio * minority_size / majority_size
        else
            target_weights[i] = 1
        end
    end

    weights = minimum(target_distribution) / target_distribution
    target_weights = [weights[x + 1] for x in target[:, 2]]
    return target_weights
end
