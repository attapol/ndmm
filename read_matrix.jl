# Utility functions for reading in sparse matrices and dense matrices
# of input and label
#
# They return two matrices: input matrix and target matrix

function read_sparse_matrix(source)
    # Read in the sparse matrix in this format
    # 
    # header: 
    # first line must be number of output units, number of input units
    # they are both indexed from zero.
    #
    # content:
    # target_index, feature1_index, feature_2_index, ...
    #
    # For example,
    #
    # 2, 23897
    # 0, 2, 4235, 1, 0, 54
    # 1, 0, 4, 14750

    f = open(source)
    num_output_units, num_input_units = int(split(readline(f),','))
    lines = readlines(f)
    I::Array{Int64,1} = zeros(length(lines) * 10000)
    J::Array{Int64,1} = zeros(length(lines) * 10000)
    target::Array{Int64, 1} = zeros(length(lines))

    current_index = 1
    for pair in enumerate(lines)
        i, line = pair
        #if i % 50 == 0
            #println(i)
        #end
        split_line = split(line, ',')
        target[i] = int(split_line[1])+1
        features = [1]
        if length(split_line) > 1
            features = [features; [int(x)+2 for x in split_line[2:]]]
            features = [1; features]
            features = features[features .< num_input_units+1]
        end
        num_features = length(features)
        I[current_index:current_index+num_features-1] = zeros(num_features) + i
        J[current_index:current_index+num_features-1] = features
        current_index += num_features
    end
    I = I[1:current_index-1]
    J = J[1:current_index-1]
    target_matrix = zeros(length(target), maximum(target))
    for i in 1:size(target_matrix, 1)
        target_matrix[i, target[i]] = 1
    end
    return sparse(I, J, 1, length(lines), num_input_units+1), target_matrix
end

function read_dense_matrix(source)
    # Read in a matrix 
    # Dense matrix does not require extra information in the header
    # But it will crash if not all rows have the same number of columns
    # the first column must be the target index
    f = open(source)
    lines = readlines(f)
    target::Array{Int64, 1} = zeros(length(lines))
    num_columns = length(split(lines[1], ',')) - 1
    num_rows = length(lines)
    data_matrix = zeros(num_rows, num_columns)
    for i in 1:num_rows
        data_vector = float(split(lines[i], ','))
        target[i] = data_vector[1]+1
        data_matrix[i, :] = data_vector[2:end]
    end

    target_matrix = zeros(length(target), maximum(target))
    for i in 1:size(target_matrix, 1)
        target_matrix[i, target[i]] = 1
    end
    return data_matrix, target_matrix
end
