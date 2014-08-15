# Module class
# include this package to access the whole library
module Net
export 
    DeepNet,
    MultiNet,
    HiddenLayer,
    OutputLayer,
    InputLayer,
    train,
    test,
    run_deep_net

# ordering is important here as julia is not really supporting OOP
include("multinet.jl")
include("activation_functions.jl")
include("read_matrix.jl")
include("intermediate_layer.jl")
include("output_layer.jl")
include("hidden_layer.jl")
include("input_layer.jl")
include("experiment_util.jl")
include("test_cases.jl")

end
