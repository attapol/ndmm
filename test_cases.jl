

function synthesize_data(num_cases, num_features) 
    # Synthesize uncorrelated data 
    # Features and labels are generated independently.
    num_output_units = 3
    data = randn(num_cases, num_features)
    target = randn(num_cases, num_output_units)

    for i in 1:num_cases
        for j in 1:num_output_units
            target[i,j] = target[i,j] == maximum(target[i,:]) ? 1 : 0
        end
    end
    return (data, target)
end

function synthesize_good_data(num_cases, num_features)
    # Synthesize the data where the features are correlated
    # with the target labels.
    num_output_units = 3
    data = randn(num_cases, num_features)
    target = randn(num_cases, num_output_units)
    weights = randn(num_features, num_output_units) * 0.01
    activations = data * weights
    for i in 1:num_cases
        for j in 1:num_output_units
            target[i,j] = activations[i,j] == maximum(activations[i,:]) ? 1 : 0
        end
    end
    return (data, target, weights)
end


function test_complex_net()
    num_cases = 20
    num_features = 10
    max_capacity = 30
    data1, target = synthesize_data(num_cases, num_features)
    data2, _ = synthesize_data(num_cases, num_features)

    net = Net.MultiNet()

    input_layer_index1 = Net.add_input_layer(net, Net.InputLayer(num_features))
    hidden_layer_index1 = Net.add_layer(net, Net.HiddenLayer(5))
    input_layer_index2 = Net.add_input_layer(net, Net.InputLayer(num_features))
    hidden_layer_index2 = Net.add_layer(net, Net.HiddenLayer(5))
    hidden_layer_index3 = Net.add_layer(net, Net.HiddenLayer(5))
    output_layer_index = Net.add_output_layer(net, Net.OutputLayer(3))

    Net.connect(net, input_layer_index1, hidden_layer_index1)
    Net.connect(net, hidden_layer_index1, output_layer_index)
    Net.connect(net, input_layer_index2, hidden_layer_index2)
    Net.connect(net, hidden_layer_index2, hidden_layer_index3)
    Net.connect(net, hidden_layer_index3, output_layer_index)
    Net.initialize(net)

    #println(net.connections)
    println(Net.compute_ordering(net))
    #Net.forward_backprop_update(net, Array[data1 data2], target, ones(num_cases))

    Net.train(net, Array[data1 data2], target, Array[data1 data2], target, 0.1, 0.0001, 40)
    return net
end

function test_one_hidden_layer(data, target, pretraining, io=STDOUT)
    num_features = size(data, 2)
    net = Net.MultiNet()
    net.io = io

    input_layer_index1 = Net.add_input_layer(net, Net.InputLayer(num_features))
    hidden_layer_index1 = Net.add_layer(net, Net.HiddenLayer(100))
    output_layer_index = Net.add_output_layer(net, Net.OutputLayer(3))
    Net.connect(net, input_layer_index1, hidden_layer_index1)
    Net.connect(net, hidden_layer_index1, output_layer_index)
    Net.initialize(net)
    Net.feed_input_layer(net, Array[data])
    
    if pretraining
        net.layers[hidden_layer_index1].learning_rate = 0.1
        Net.pretrain(net.layers[hidden_layer_index1], 5)
    end

    Net.initialize(net)
    Net.train(net, Array[data], target, Array[data], target, 0.01, 0.0001, 40)
    return Net.test(net, Array[data], target)
end

function test_simple_net(data, target, io=STDOUT)
    num_features = size(data, 2)
    net = Net.MultiNet()
    net.io = io

    input_layer_index1 = Net.add_input_layer(net, Net.InputLayer(num_features))
    output_layer_index = Net.add_output_layer(net, Net.OutputLayer(3))
    Net.connect(net, input_layer_index1, output_layer_index)

    Net.initialize(net)
    Net.train(net, Array[data], target, Array[data], target, 0.1, 0.0001, 40)
    return Net.test(net, Array[data], target)
end

function test_xor(io=STDOUT)
    # hidden layer is required for this
    # it's surprisingly hard to get this to work without online learning...
    data = [
        0 0 ;
        0 1 ;
        1 0 ;
        1 1 
        ]
    target = [ 
        1 0 ;
        0 1 ;
        0 1 ;
        1 0 
        ]
    net1 = Net.MultiNet()
    net1.io = io
    input_layer_index1 = Net.add_input_layer(net1, Net.InputLayer(2))
    output_layer_index = Net.add_output_layer(net1, Net.OutputLayer(2))
    Net.connect(net1, input_layer_index1, output_layer_index)
    Net.initialize(net1)
    Net.train(net1, Array[data], target, Array[data], target, 0.1, 0.0001, 100)
    _, accuracy1 = Net.test(net1, Array[data], target)

    net2 = Net.MultiNet()
    net2.io = io
    input_layer_index1 = Net.add_input_layer(net2, Net.InputLayer(2))
    hidden_layer_index1 = Net.add_layer(net2, Net.HiddenLayer(2))
    output_layer_index = Net.add_output_layer(net2, Net.OutputLayer(2))
    net2.layers[hidden_layer_index1].learning_rate = 0.01
    net2.layers[output_layer_index].learning_rate = 0.01

    Net.connect(net2, input_layer_index1, hidden_layer_index1)
    Net.connect(net2, hidden_layer_index1, output_layer_index)
    Net.initialize(net2)


    Net.train(net2, Array[data], target, Array[data], target, 0.1, 0.0001, 500)
    _, accuracy2 = Net.test(net2, Array[data], target)
    return (accuracy1, accuracy2)
    #return (net1, net2)
end

function test_mixture_of_experts(io=STDOUT)
    num_cases = 100
    num_features = 10
    max_capacity = 30
    data, _ = synthesize_data(num_cases, num_features)
    data += 3
    good_data, target, true_weights = synthesize_good_data(num_cases, num_features)

    net = Net.MultiNet()

    input_layer_index1 = Net.add_input_layer(net, Net.InputLayer(num_features))
    hidden_layer_index1 = Net.add_layer(net, Net.HiddenLayer(5))
    output_layer_index1 = Net.add_output_layer(net, Net.OutputLayer(3))

    input_layer_index2 = Net.add_input_layer(net, Net.InputLayer(num_features))
    hidden_layer_index2 = Net.add_layer(net, Net.HiddenLayer(5))
    output_layer_index2 = Net.add_output_layer(net, Net.OutputLayer(3))

    Net.connect(net, input_layer_index1, hidden_layer_index1)
    Net.connect(net, hidden_layer_index1, output_layer_index1)

    Net.connect(net, input_layer_index2, hidden_layer_index2)
    Net.connect(net, hidden_layer_index2, output_layer_index2)

    gated_output_layer = Net.GatedOutputLayer(3)
    Net.add_expert(gated_output_layer, net.layers[output_layer_index1])
    Net.add_expert(gated_output_layer, net.layers[output_layer_index2])

    gated_output_layer_index = Net.add_output_layer(net, gated_output_layer)
    Net.connect(net, input_layer_index1, gated_output_layer_index)
    Net.connect(net, input_layer_index2, gated_output_layer_index)
    Net.initialize(net)


    println(Net.compute_ordering(net))

    Net.train(net, Array[data, good_data], target, Array[data, good_data], target, 0.1, 0.000001, 25, 20)
    return net
    
end
