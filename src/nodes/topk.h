/* This file is part of onnx2c.
 *
 * Specialised implementation of the TopK node 
 * strictly for axis = -1, largest = 1, sorted = 1.
 */
#include "node.h"

namespace toC {

class TopK : public Node {
    public:
    TopK() {
        op_name = "TopK";
    }

    // Hardcoded attributes as requested
    const int axis_ = -1;
    const int largest_ = 1;
    const int sorted_ = 1;
    
    int64_t k_value_ = 1; 

    virtual void parseAttributes( onnx::NodeProto &node ) override;
    virtual void resolve(void) override;
    virtual void print(std::ostream &dst) const override;
};


/* Parse attributes and enforce strict constraints */
void TopK::parseAttributes( onnx::NodeProto &node )
{
    for( const auto& a : node.attribute() ) {
        if( a.name() == "axis" ) {
            int parsed_axis = parse_attribute_int(a);
            if (parsed_axis != -1) {
                LOG(ERROR) << "Strict TopK: unsupported axis " << parsed_axis << ". Only axis=-1 is allowed." << std::endl;
                exit(1);
            }
        }
        else if( a.name() == "largest" ) {
            int parsed_largest = parse_attribute_int(a);
            if (parsed_largest != 1) {
                LOG(ERROR) << "Strict TopK: unsupported largest=" << parsed_largest << ". Only largest=1 is allowed." << std::endl;
                exit(1);
            }
        }
        else if( a.name() == "sorted" ) {
            int parsed_sorted = parse_attribute_int(a);
            if (parsed_sorted != 1) {
                LOG(ERROR) << "Strict TopK: unsupported sorted=" << parsed_sorted << ". Only sorted=1 is allowed." << std::endl;
                exit(1);
            }
        }
    }
}


/* Assign input tensors, resolve output tensor shapes, allocate output tensors */
void TopK::resolve(void)
{
    Tensor *X = get_input_tensor(0);
    name_input(0, "X");

    Tensor *K_tensor = get_input_tensor(1);
    name_input(1, "K");

    int rank = X->data_dim.size();
    int actual_axis = rank - 1; // axis is strictly -1

    // Estrazione sicura del parametro K tramite casting del data_buffer
    if (!K_tensor->isConst || K_tensor->data_buffer == nullptr) {
        LOG(ERROR) << "Strict TopK: Il tensore K deve essere una costante (isConst=true)!" << std::endl;
        exit(1);
    }

    if (K_tensor->data_type == onnx::TensorProto_DataType_INT64) {
        int64_t* k_array = static_cast<int64_t*>(K_tensor->data_buffer);
        k_value_ = k_array[0];
    } 
    else if (K_tensor->data_type == onnx::TensorProto_DataType_INT32) {
        int32_t* k_array = static_cast<int32_t*>(K_tensor->data_buffer);
        k_value_ = k_array[0];
    } 
    else {
        LOG(ERROR) << "Strict TopK: Il tipo di dato per K non è supportato. Deve essere INT64 o INT32." << std::endl;
        exit(1);
    }

    /* Output 1: Values */
    Tensor *Values = new Tensor;
    Values->data_dim = X->data_dim;
    Values->data_dim[actual_axis] = k_value_;
    Values->data_type = X->data_type;
    register_output(Values, "Values");

    /* Output 2: Indices */
    Tensor *Indices = new Tensor;
    Indices->data_dim = X->data_dim;
    Indices->data_dim[actual_axis] = k_value_;
    Indices->data_type = onnx::TensorProto_DataType_INT64;
    register_output(Indices, "Indices");
}


/* Body of the node implementing function - Highly optimized for axis=-1 */
void TopK::print(std::ostream &dst) const
{
    Tensor *X = get_input_tensor(0);
    int rank = X->data_dim.size();
    
    INDT_1 << "/* Optimized TopK (axis=-1, largest=1, sorted=1) */" << std::endl;
    
    // axis_size is simply the last dimension
    int axis_size = X->data_dim[rank - 1];
    
    // outer_size is the product of all dimensions EXCEPT the last one
    int outer_size = 1;
    for (int i = 0; i < rank - 1; ++i) {
        outer_size *= X->data_dim[i];
    }

    std::string c_type = X->data_type == onnx::TensorProto_DataType_FLOAT ? "float" : "int64_t";

    INDT_1 << "int outer_size = " << outer_size << ";" << std::endl;
    INDT_1 << "int axis_size = " << axis_size << ";" << std::endl;
    INDT_1 << "int k_val = " << k_value_ << ";" << std::endl;
    
    // We only need two nested loops now: Outer and K
    INDT_1 << "for (int o = 0; o < outer_size; ++o) {" << std::endl;
    
    // Base offsets are much simpler without inner_size
    INDT_2 << "int base_offset = o * axis_size;" << std::endl;
    INDT_2 << "int out_base_offset = o * k_val;" << std::endl;

    INDT_2 << "int selected[axis_size];" << std::endl;
    INDT_2 << "for (int j = 0; j < axis_size; ++j) selected[j] = 0;" << std::endl;

    INDT_2 << "for (int k = 0; k < k_val; ++k) {" << std::endl;
    INDT_3 << c_type << " best_val;" << std::endl;
    INDT_3 << "int64_t best_idx = -1;" << std::endl;
    INDT_3 << "int found = 0;" << std::endl;

    INDT_3 << "for (int j = 0; j < axis_size; ++j) {" << std::endl;
    INDT_4 << "if (selected[j]) continue;" << std::endl;
    INDT_4 << c_type << " val = X[base_offset + j];" << std::endl;
    // Hardcoded logic for largest=1 (finding the maximum)
    INDT_4 << "if (!found || val > best_val) {" << std::endl;
    INDT_5 << "best_val = val;" << std::endl;
    INDT_5 << "best_idx = j;" << std::endl;
    INDT_5 << "found = 1;" << std::endl;
    INDT_4 << "}" << std::endl;
    INDT_3 << "}" << std::endl;

    INDT_3 << "if (best_idx != -1) {" << std::endl;
    INDT_4 << "selected[best_idx] = 1;" << std::endl;
    INDT_4 << "Values[out_base_offset + k] = best_val;" << std::endl;
    INDT_4 << "Indices[out_base_offset + k] = best_idx;" << std::endl;
    INDT_3 << "}" << std::endl;
    
    INDT_2 << "}" << std::endl; // end for k
    INDT_1 << "}" << std::endl; // end for outer
}

} // namespace toC