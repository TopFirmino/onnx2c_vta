/* This file is part of onnx2c.
 *
 * TEMPLATE node.
 * When implementing a new node, use this template
 * as a starting point.
 *
 * This file can be kept as a single .h file with an
 * in-header implementation, or it can be split into
 * a .h and a .cc file.
 *
 * Replace all occurances of TEMPLATE in this file.
 * Some representative dummy implementation provided.
 *
 * The functions here are callbacks from the onnx2c
 * framework. See node.h for more documentation.
 */
#include "node.h"

namespace toC {

class VTANode : public Node {
	public:
	VTANode() {
		op_name = "VTANode";
	}
    std::string program_name;
    uint32_t output_count;

	// Mandatory "API" functions towards the rest of onnx2c
	virtual void parseAttributes( onnx::NodeProto &node ) override;
	virtual void resolve(void) override;
	virtual void print(std::ostream &dst) const override;
};

/* Parse attributes, if this node has them. */
void VTANode::parseAttributes( onnx::NodeProto &node )
{
	for( const auto& a : node.attribute() ) {
		LOG(TRACE) << "Parsing attribute " << a.name() << std::endl;
		if( a.name() == "program_name" )
			program_name = parse_attribute_string(a);
        else if (a.name() == "output_count") {
            output_count = parse_attribute_int(a);
        } else
			LOG(ERROR) << "Ignoring attribute " << a.name() << " for node VTANode/" << onnx_name << std::endl;
	}
}

/* Assign input tensors, resolve output tensor shapes, allocate output tensors */
void VTANode::resolve(void)
{
    for (size_t i = 0; i < get_number_of_inputs(); i++) {
        // Tensor *input  = get_input_tensor(i);
        name_input(i, "VTA_T" + std::to_string(i+1));
    }
    

	/* Create output tensors.
	 * Set data dimensions and data type for the created tensors. */
    for (size_t i = 0; i < output_count; i++) {
        Tensor *t = new Tensor;
        t->data_dim.push_back(1);  // Just as placeholder, the VTA_Get_Tensor node specify the exact dim
        t->data_type = onnx::TensorProto_DataType_UINT32;
        register_output(t, "Y" + std::to_string(i+1));
    }
}

/* Body of the node implementing function */
void VTANode::print(std::ostream &dst) const
{
	INDT_1 << "/* Print info on this node here, for debugging purposes */" << std::endl;

	/* Genereate the C code here */
	INDT_2 << "VTARunProgram('" << program_name << "');" << std::endl;
}


} // namespace



namespace toC {

class VTAExtract : public Node {
	public:
	VTAExtract() {
		op_name = "VTA_Extract";
	}
    std::string variable_name;
	std::string program_name;
    std::vector<int64_t> shape;

	// Mandatory "API" functions towards the rest of onnx2c
	virtual void parseAttributes( onnx::NodeProto &node ) override;
	virtual void resolve(void) override;
	virtual void print(std::ostream &dst) const override;
};

/* Parse attributes, if this node has them. */
void VTAExtract::parseAttributes( onnx::NodeProto &node )
{
	for( const auto& a : node.attribute() ) {
		LOG(TRACE) << "Parsing attribute " << a.name() << std::endl;
		if( a.name() == "variable_name" )
			variable_name = parse_attribute_string(a);
		if( a.name() == "program_name" )
			program_name = parse_attribute_string(a);
        else if( a.name() == "shape" )
			shape = parse_attribute_ints(a);
        else
			LOG(ERROR) << "Ignoring attribute " << a.name() << " for node VTA_Extract/" << onnx_name << std::endl;
	}
}

/* Assign input tensors, resolve output tensor shapes, allocate output tensors */
void VTAExtract::resolve(void)
{
    assert(get_number_of_inputs() == 1);
    name_input(0, "inp");
    
    Tensor *t = new Tensor;
    for (auto s : shape)
        t->data_dim.push_back(s);
    t->data_type = onnx::TensorProto_DataType_INT8;
    register_output(t, "y");
}

/* Body of the node implementing function */
void VTAExtract::print(std::ostream &dst) const
{
	INDT_1 << "/* Print info on this node here, for debugging purposes */" << std::endl;

	/* Genereate the C code here */
	INDT_2 << "VTACopyToHost('" << program_name << "', '" << variable_name << "');" << std::endl;
	INDT_2 << "Reshape();";
}


} // namespace




