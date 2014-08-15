# Run multi-layer (deep) neural network experiments
#
# Example use: Run the experiment on 4 processes
# julia -p 4 train_tune_multideepnet.jl test_tune_config.json
#
#
# Here are the fields that the json config file is taking.
# Look at test_tune_config.json as an example
#
# I/O CONTROL 
# save_model : saving the model at the end of the experiment or not
# log_to_file : logging the STDOUT to a file or not
#
# CLASSIFICATION TASK FORMULATION
# multiclass : running multiclass classification or not (default: true)
# label_index_list : running one-vs-all classification on these label indices.
#                    ignored if multiclass = true
#
# MODEL TRAINING ENVIRONMENT
# num_repeats : number of repetition (new random starts) (default : 1)
# num_iterations : number of iterations
# pretraining_max_iterations : the maximum iterations of pretraining. set to zero to turn off
# 
# HYPERPARAMETERS TO TUNE.  We will run all combinations of these hyperparameters
# num_hidden_unit_list : a list of number of hidden units to try. All hidden layers have the same
# activation_fn_list : a list of activation functions to try. options are "relu" and "sigmoid"
# learning_rate_list : a list of AdaGrad meta-learning rates to try
# lr_smoother_list : a list of AdaGrad denominator smoothing constant to try
# batch_size_list : a list of mini batch sizes to try
# reweighting_factor_list : a list of weighting factors to try. between 0 and 1
#       if 1, the majority class gets downweighted the most. if 0, no reweighting.
#
# ARCHITECTURE SPECIFICATION
# architecture : a list of layer specifcations. Each layer is indexed by the order it appears.
# 
#   Layer specifications (a dictionary)
#   - Input Layer 
#       type : "input"
#       parents : should be empty []
#       data_sparse or data_dense : sparse or dense feature vectors (see data format)
#   - Hidden Layer 
#       type : "hidden" - the number of hidden units will vary based on num_hidden_unit_list
#              "hidden_fixed" - the number of hidden units is fixed 
#       parents : a list of parent indices
#       num_units : used only for "hidden_fixed" type. 
#   - Output Layer
#       type : "output"
#       parents : same
#   - Gated output layers : for Mixture of Experts architecture
#       type : "gated_output"
#       parents : a list of input or hidden layers that help determine the gating weights
#       experts : a list of output layers (from the experts)
#

using Net
using JSON
using Iterators

function main(args)
    config_file = args[1]
    config_dict = JSON.parse(readall(open(config_file)))
    if !haskey(config_dict, "file_name_prefix")
        config_dict["file_name_prefix"] = splitext(config_file)[1]
    end
    srand(10)
    num_repeats = Net.get_config_value(config_dict, "num_repeats", 1)
    multiclass = Net.get_config_value(config_dict, "multiclass", true)

    header = "num_hidden_unit, activation_fn, learning_rate, lr_smoother, batch_size, reweighting_factor, label_index, loss, accuracy, mean_precision, mean_recall, mean_f1"
    # get all combinations of parameters 
    parameters = {}
    for combination in Iterators.product(
                config_dict["num_hidden_unit_list"], 
                config_dict["activation_fn_list"],
                config_dict["learning_rate_list"], 
                config_dict["lr_smoother_list"], 
                config_dict["batch_size_list"], 
                config_dict["reweighting_factor_list"],
                1:num_repeats)
        if multiclass
            parameter_set = (deepcopy(config_dict), combination, -1)
            push!(parameters, parameter_set)
        else
            label_index_list = config_dict["label_index_list"]
            for label_index in label_index_list
                parameter_set = (deepcopy(config_dict), combination, label_index)
                push!(parameters, parameter_set)
            end
        end
    end

    output_file = open("$(config_dict["file_name_prefix"]).csv", "w")

    # run them in parallel
    results = parmap(do_fn, parameters, output_file)
    println(length(parameters))
    println(length(results))
end

function summarize_confusion_matrix(confusion_matrix, io)
    num_classes = size(confusion_matrix, 1)
    accuracy = trace(confusion_matrix) / sum(confusion_matrix)
    precisions = [confusion_matrix[i,i] / sum(confusion_matrix[:,i]) for i in 1:size(confusion_matrix,1)]'
    recalls = [confusion_matrix[i,i] / sum(confusion_matrix[i,:]) for i in 1:size(confusion_matrix,1)]'
    f1s = 2 * (precisions .* recalls) ./ (precisions + recalls) 

    mean_f1 = mean(f1s)
    mean_precision = mean(precisions)
    mean_recall = mean(recalls)
    if accuracy == NaN
        println(confusion_matrix)
    end
    print(io, ", $accuracy, $(mean(precisions)), $(mean(recalls)), $(mean(f1s))")
    for i in 1:num_classes
        print(io, ", $(precisions[i]), $(recalls[i]), $(f1s[i])")
    end
    return (accuracy, precisions, recalls, f1s)
end

function write_result(io, losses, dev_confusion_matrices, test_confusion_matrices, parameters)
    config_dict, hyperparameters, label_index = parameters
    num_hidden_unit, activation_fn, learning_rate, lr_smoother, batch_size, reweighting_factor, repeat = hyperparameters
    write_header = position(io) == 0
    has_dev_set = dev_confusion_matrices[1] != None
    has_test_set = test_confusion_matrices[1] != None

    confusion_matrix = has_dev_set ? dev_confusion_matrices[1] : test_confusion_matrices[1]
    num_classes = size(confusion_matrix, 1)
    # writer header of the csv file
    if write_header
        header = "num_hidden_unit, activation_fn, learning_rate, lr_smoother, batch_size, reweighting_factor, label_index, repeat, iteration, loss" 
        if has_dev_set
            header *= ", dev_accuracy, dev_mean_precision, dev_mean_recall, dev_mean_f1"
            for i in 1:num_classes
                header *= ", dev_precision_class$i, dev_recall_class$i, dev_f1_class$i"
            end
        end
        if has_test_set
            header *= ", test_accuracy, test_mean_precision, test_mean_recall, test_mean_f1"
            for i in 1:num_classes
                header *= ", test_precision_class$i, test_recall_class$i, test_f1_class$i"
            end
        end
        println(io, header)
    end

    #fill in the results
    for j in 1:length(losses)
        loss = losses[j]
        print(io, "$num_hidden_unit, $activation_fn, $learning_rate, $lr_smoother, $batch_size, $reweighting_factor, $label_index, $repeat, $j, $loss")
        if has_dev_set
            summarize_confusion_matrix(dev_confusion_matrices[j], io)
        end
        if has_test_set
            summarize_confusion_matrix(test_confusion_matrices[j], io)
        end
        println(io,"")
        flush(io)
    end
end

function parmap(f, lst, output_file)
    np = nprocs()  # determine the number of processes available
    n = length(lst)
    results = cell(n)
    i = 1
    # function to produce the next work item from the queue.
    # in this case it's just an index.
    nextidx() = (idx=i; i+=1; idx)
    @sync begin
        for p=1:np
            sleep(1)
            if p != myid() || np == 1
                @async begin
                    while true
                        idx = nextidx()
                        if idx > n
                            break
                        end
                        results[idx] = remotecall_fetch(p, f, lst[idx])
                        losses, dev_confusion_matrices, test_confusion_matrices = results[idx]
                        write_result(output_file, losses, 
                            dev_confusion_matrices, test_confusion_matrices, lst[idx])
                    end
                end
            end
        end
    end
    results
end

@everywhere function do_fn(parameters)
    config_dict, hyperparameters, label_index = parameters
    num_hidden_unit, activation_fn, learning_rate, lr_smoother, batch_size, reweighting_factor, repeat = hyperparameters

    config_dict["learning_rate"] = learning_rate
    config_dict["lr_smoother"] = lr_smoother
    config_dict["minibatch_size"] = batch_size
    config_dict["activation_fn"] = activation_fn
    config_dict["reweighting_factor"] = reweighting_factor
    layers = config_dict["architecture"]
    for layer in layers
        if layer["type"] == "hidden"
            layer["num_units"] = num_hidden_unit
        end
    end
    losses, dev_confusion_matrices, test_confusion_matrices = Net.run_multinet(label_index, config_dict)
    return (losses, dev_confusion_matrices, test_confusion_matrices) 
end

main(ARGS)
