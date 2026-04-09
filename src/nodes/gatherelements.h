/* This file is part of onnx2c.
 *
 * Implementation of the GatherElements node.
 */
#include "node.h"
#include <string>

namespace toC {

class GatherElements : public Node {
    public:
    GatherElements() {
        op_name = "GatherElements";
    }

    // L'attributo 'axis' ha default 0 in ONNX
    int axis_ = 0;

    virtual void parseAttributes( onnx::NodeProto &node ) override;
    virtual void resolve(void) override;
    virtual void print(std::ostream &dst) const override;
};


/* Parsa l'attributo axis */
void GatherElements::parseAttributes( onnx::NodeProto &node )
{
    for( const auto& a : node.attribute() ) {
        if( a.name() == "axis" ) {
            axis_ = parse_attribute_int(a);
        } else {
            LOG(WARNING) << "GatherElements: Ignored attribute " << a.name() << std::endl;
        }
    }
}


/* Valida gli input e calcola la forma dell'output */
void GatherElements::resolve(void)
{
    Tensor *data = get_input_tensor(0);
    name_input(0, "data");

    Tensor *indices = get_input_tensor(1);
    name_input(1, "indices");

    int rank = data->data_dim.size();

    // ONNX richiede che data e indices abbiano lo stesso rango (r >= 1)
    if ((int)indices->data_dim.size() != rank || rank < 1) {
        LOG(ERROR) << "GatherElements: 'data' e 'indices' devono avere lo stesso rango, e rank >= 1." << std::endl;
        exit(1);
    }

    // Risolviamo l'asse negativo (-rank <= axis < rank)
    int actual_axis = axis_ < 0 ? rank + axis_ : axis_;
    if (actual_axis < 0 || actual_axis >= rank) {
        LOG(ERROR) << "GatherElements: Asse fuori dai limiti consentiti." << std::endl;
        exit(1);
    }

    /* Output Y ha esattamente la stessa forma del tensore 'indices' */
    Tensor *Y = new Tensor;
    Y->data_dim = indices->data_dim; 
    Y->data_type = data->data_type;  // Il tipo di dato dell'output è uguale a quello del tensor 'data'
    register_output(Y, "Y");
}


/* Generazione codice C: cicli nidificati e risoluzione offset a runtime */
void GatherElements::print(std::ostream &dst) const
{
    Tensor *data = get_input_tensor(0);
    Tensor *indices = get_input_tensor(1);
    
    int rank = data->data_dim.size();
    int actual_axis = axis_ < 0 ? rank + axis_ : axis_;

    INDT_1 << "/* GatherElements Implementation (axis=" << actual_axis << ") */" << std::endl;

    // Calcolo degli stride (salti in memoria) per data e indices/Y
    std::vector<int> data_strides(rank, 1);
    std::vector<int> idx_strides(rank, 1);
    
    for (int i = rank - 2; i >= 0; --i) {
        data_strides[i] = data_strides[i+1] * data->data_dim[i+1];
        idx_strides[i]  = idx_strides[i+1]  * indices->data_dim[i+1];
    }

    auto indent = [](int level) { return std::string(level * 4, ' '); };

    // 1. Generiamo i cicli FOR dinamici sulla base delle dimensioni di 'indices'
    for (int i = 0; i < rank; ++i) {
        dst << indent(1 + i) << "for (int d" << i << " = 0; d" << i << " < " << indices->data_dim[i] << "; ++d" << i << ") {\n";
    }

    int inner_level = rank + 1;

    // 2. Costruiamo la stringa per calcolare l'indice flat di 'indices' (e di Y)
    std::string flat_idx_str = "";
    for (int i = 0; i < rank; ++i) {
        flat_idx_str += "d" + std::to_string(i) + " * " + std::to_string(idx_strides[i]);
        if (i < rank - 1) flat_idx_str += " + ";
    }

    // Leggiamo l'indice raw dal tensore indices
    dst << indent(inner_level) << "long raw_idx = indices[" << flat_idx_str << "];\n";
    
    // Supporto per indici negativi richiesto da ONNX (es: -1 diventa dim - 1)
    dst << indent(inner_level) << "long actual_idx = raw_idx < 0 ? raw_idx + " << data->data_dim[actual_axis] << " : raw_idx;\n";

    // 3. Costruiamo l'indice flat per accedere a 'data'
    // La particolarità di GatherElements è che lungo l'asse "axis", 
    // la coordinata è data dal valore letto da 'indices' (actual_idx),
    // mentre per tutte le altre dimensioni si mantiene la coordinata corrente (d0, d1, ecc.)
    std::string flat_data_str = "";
    for (int i = 0; i < rank; ++i) {
        if (i == actual_axis) {
            flat_data_str += "actual_idx * " + std::to_string(data_strides[i]);
        } else {
            flat_data_str += "d" + std::to_string(i) + " * " + std::to_string(data_strides[i]);
        }
        if (i < rank - 1) flat_data_str += " + ";
    }

    // Eseguiamo la copia del dato
    dst << indent(inner_level) << "Y[" << flat_idx_str << "] = data[" << flat_data_str << "];\n";

    // Chiudiamo tutti i cicli FOR
    for (int i = rank - 1; i >= 0; --i) {
        dst << indent(1 + i) << "}\n";
    }
}

} // namespace toC