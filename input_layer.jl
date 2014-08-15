#Input Layer implementation
#
# InputLayer is there to hold the input 

type InputLayer <: Layer
    num_units::Int64
    data_range
    learning_rate::Float64
    lr_smoother::Float64
    activations
    incoming_backprop_delta

    InputLayer(num_units::Int64) = begin
       return new(num_units, None, 0.0, 0.0, [], []) 
    end
end

function get_activations(this::InputLayer)
    return this.data_range==None ? this.activations : this.activations[this.data_range, :]
end

function feed_forward(this::InputLayer)
end

function update(this::InputLayer, target=None, target_weights=None)
end

function initialize(this::InputLayer)
end
