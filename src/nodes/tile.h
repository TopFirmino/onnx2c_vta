/* This file is part of onnx2c.
 *
 * Implementation of the Tile node.
 */
#include "node.h"
#include <string>

namespace toC {

class Tile : public Node {
    public:
    Tile() {
        op_name = "Tile";
    }

    // L'operatore Tile non ha attributi in ONNX.
    // Salviamo internamente l'array dei 'repeats'.
    std::vector<int64_t> repeats_;

    virtual void parseAttributes( onnx::NodeProto &node ) override;
    virtual void resolve(void) override;
    virtual void print(std::ostream &dst) const override;
};


/* Tile non possiede attributi nativi (tutto passa via input tensor) */
void Tile::parseAttributes( onnx::NodeProto &node )
{
    if (node.attribute_size() > 0) {
        LOG(WARNING) << "Il nodo Tile non dovrebbe avere attributi secondo le specifiche ONNX." << std::endl;
    }
}


/* Assegna gli input, valida i 'repeats' e calcola la forma dell'output */
void Tile::resolve(void)
{
    Tensor *X = get_input_tensor(0);
    name_input(0, "X");

    Tensor *repeats_tensor = get_input_tensor(1);
    name_input(1, "repeats");

    int rank = X->data_dim.size();

    // In onnx2c il tensore dei repeats DEVE essere noto a tempo di compilazione.
    if (!repeats_tensor->isConst || repeats_tensor->data_buffer == nullptr) {
        LOG(ERROR) << "Tile: Il tensore 'repeats' deve essere una costante inizializzata (isConst=true)!" << std::endl;
        exit(1);
    }

    // Estrazione dei valori di repeats
    repeats_.resize(rank);
    if (repeats_tensor->data_type == onnx::TensorProto_DataType_INT64) {
        int64_t* rep_array = static_cast<int64_t*>(repeats_tensor->data_buffer);
        for(int i = 0; i < rank; ++i) repeats_[i] = rep_array[i];
    } 
    else if (repeats_tensor->data_type == onnx::TensorProto_DataType_INT32) {
        int32_t* rep_array = static_cast<int32_t*>(repeats_tensor->data_buffer);
        for(int i = 0; i < rank; ++i) repeats_[i] = rep_array[i];
    } 
    else {
        LOG(ERROR) << "Tile: Il tipo di dato per 'repeats' deve essere INT64 o INT32." << std::endl;
        exit(1);
    }

    /* Creazione del tensore di Output (Y) */
    Tensor *Y = new Tensor;
    for (int i = 0; i < rank; ++i) {
        // La dimensione finale è la dimensione originale moltiplicata per il numero di ripetizioni
        Y->data_dim.push_back(X->data_dim[i] * repeats_[i]);
    }
    Y->data_type = X->data_type;
    register_output(Y, "Y");
}


/* Generazione del codice C - Crea dinamicamente N cicli FOR in base al rango */
void Tile::print(std::ostream &dst) const
{
    Tensor *X = get_input_tensor(0);
    int rank = X->data_dim.size();

    INDT_1 << "/* Tile Implementation */" << std::endl;

    if (rank == 0) {
        // Edge case: scalare puro (nessuna dimensione)
        INDT_1 << "Y[0] = X[0];" << std::endl;
        return;
    }

    // Calcolo degli stride (passi di memoria) per X e per Y
    std::vector<int> x_strides(rank, 1);
    std::vector<int> y_strides(rank, 1);
    for (int i = rank - 2; i >= 0; --i) {
        x_strides[i] = x_strides[i+1] * X->data_dim[i+1];
        y_strides[i] = y_strides[i+1] * (X->data_dim[i+1] * repeats_[i+1]);
    }

    // Helper per l'indentazione dinamica del codice C generato
    auto indent = [](int level) { return std::string(level * 4, ' '); };

    // Generazione dei cicli FOR annidati
    for (int i = 0; i < rank; ++i) {
        int dim_out_size = X->data_dim[i] * repeats_[i];
        dst << indent(1 + i) << "for (int d" << i << " = 0; d" << i << " < " << dim_out_size << "; ++d" << i << ") {\n";
    }

    // Costruzione delle equazioni di accesso lineare alla memoria (flat indexing)
    std::string x_idx = "";
    std::string y_idx = "";
    for (int i = 0; i < rank; ++i) {
        // Magia del Tile: l'indice in X usa l'operatore modulo (%) sulla dimensione originale di X!
        x_idx += "(d" + std::to_string(i) + " % " + std::to_string(X->data_dim[i]) + ") * " + std::to_string(x_strides[i]);
        y_idx += "d" + std::to_string(i) + " * " + std::to_string(y_strides[i]);
        if (i < rank - 1) {
            x_idx += " + ";
            y_idx += " + ";
        }
    }

    // Corpo centrale: copia il valore
    dst << indent(1 + rank) << "Y[" << y_idx << "] = X[" << x_idx << "];\n";

    // Chiusura di tutti i cicli FOR
    for (int i = rank - 1; i >= 0; --i) {
        dst << indent(1 + i) << "}\n";
    }
}

} // namespace toC