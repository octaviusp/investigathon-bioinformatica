/**
 * @file data_cleaning_pipeline.cpp
 * @brief Data Cleaning Pipeline para MIDORI2 COI Dataset
 *
 * Pipeline de limpieza de datos que procesa el dataset MIDORI2 de secuencias
 * COI (Cytochrome Oxidase I) para clasificación taxonómica.
 *
 * Funcionalidades:
 *   1. Elimina secuencias duplicadas usando hash de contenido
 *   2. Genera 6 datasets independientes (uno por nivel taxonómico)
 *   3. Filtra outliers por longitud dentro de cada grupo taxonómico
 *      usando el criterio: media ± std/2
 *
 * Compilar:
 *   g++ -O3 -std=c++17 -o data_cleaning_pipeline data_cleaning_pipeline.cpp
 *
 * Ejecutar:
 *   ./data_cleaning_pipeline
 *
 * Input:
 *   - data/MIDORI2_UNIQ_NUC_GB268_CO1.taxon  (~1.7M registros)
 *   - data/MIDORI2_UNIQ_NUC_GB268_CO1.fasta  (~1.7M secuencias)
 *
 * Output:
 *   - data/data_clean_{level}.csv   (6 archivos)
 *   - data/data_clean_{level}.fasta (6 archivos)
 *   donde level = {species, genus, family, order, class, phylum}
 *
 * @author Generated with Claude Code
 * @date 2024
 */

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace std;

// =============================================================================
// CONFIGURACIÓN
// =============================================================================

const string TAXON_FILE = "data/MIDORI2_UNIQ_NUC_GB268_CO1.taxon";
const string FASTA_FILE = "data/MIDORI2_UNIQ_NUC_GB268_CO1.fasta";

/** Niveles taxonómicos a procesar */
struct TaxLevel {
    string name;        // Nombre interno
    string file_suffix; // Sufijo para archivos de salida
};

const vector<TaxLevel> TAX_LEVELS = {
    {"species", "species"}, {"genus", "genus"},   {"family", "family"},
    {"order", "order"},     {"class_", "class"},  {"phylum", "phylum"}
};

// =============================================================================
// ESTRUCTURAS DE DATOS
// =============================================================================

/** Registro de una secuencia con su taxonomía */
struct Record {
    string seq_id;
    int length;
    string kingdom;
    string phylum;
    string class_;
    string order;
    string family;
    string genus;
    string species;
    size_t seq_hash;
    bool valid;
};

// =============================================================================
// FUNCIONES DE UTILIDAD
// =============================================================================

/** Calcula hash rápido de una secuencia para detección de duplicados */
size_t fast_hash(const string &str) {
    return hash<string>{}(str);
}

/**
 * Extrae la longitud de la secuencia del ID.
 *
 * Soporta dos formatos de ID:
 *   - "ACC.<start.>end"  -> length = end - start + 1
 *   - "ACC.start.end"    -> length = end - start + 1
 *
 * @param seq_id ID de la secuencia
 * @return Longitud calculada, o -1 si no se puede parsear
 */
int extract_length(const string &seq_id) {
    smatch m;

    // Formato 1: <start.>end
    regex r1("<(\\d+)\\.>(\\d+)");
    if (regex_search(seq_id, m, r1)) {
        return stoi(m[2]) - stoi(m[1]) + 1;
    }

    // Formato 2: .start.end al final
    regex r2("\\.(\\d+)\\.(\\d+)$");
    if (regex_search(seq_id, m, r2)) {
        return stoi(m[2]) - stoi(m[1]) + 1;
    }

    return -1;
}

/** Elimina el ID numérico del final del nombre taxonómico */
string clean_tax_name(const string &name) {
    return regex_replace(name, regex("_\\d+$"), "");
}

/** Parsea string de taxonomía en diccionario {prefijo: nombre} */
unordered_map<string, string> parse_taxonomy(const string &taxonomy) {
    unordered_map<string, string> tax_dict;
    stringstream ss(taxonomy);
    string token;

    while (getline(ss, token, ';')) {
        size_t pos = token.find("__");
        if (pos != string::npos) {
            tax_dict[token.substr(0, pos)] = clean_tax_name(token.substr(pos + 2));
        }
    }
    return tax_dict;
}

/** Obtiene el valor del nivel taxonómico para un registro */
string get_tax_level(const Record &r, const string &level) {
    if (level == "species") return r.species;
    if (level == "genus") return r.genus;
    if (level == "family") return r.family;
    if (level == "order") return r.order;
    if (level == "class_") return r.class_;
    if (level == "phylum") return r.phylum;
    return "";
}

// =============================================================================
// FUNCIONES DE PARSING
// =============================================================================

/** Lee y parsea el archivo .taxon */
vector<Record> parse_taxon(const string &filepath) {
    cout << "Leyendo taxon: " << filepath << endl;
    vector<Record> records;
    ifstream file(filepath);
    string line;

    while (getline(file, line)) {
        if (line.empty()) continue;

        size_t tab_pos = line.find('\t');
        if (tab_pos == string::npos) continue;

        string seq_id = line.substr(0, tab_pos);
        string taxonomy = line.substr(tab_pos + 1);

        int length = extract_length(seq_id);
        if (length <= 0 || length > 5000) continue;

        auto tax = parse_taxonomy(taxonomy);

        Record r;
        r.seq_id = seq_id;
        r.length = length;
        r.kingdom = tax["k"];
        r.phylum = tax["p"];
        r.class_ = tax["c"];
        r.order = tax["o"];
        r.family = tax["f"];
        r.genus = tax["g"];
        r.species = tax["s"];
        r.valid = true;

        records.push_back(r);
    }

    cout << "  Registros parseados: " << records.size() << endl;
    return records;
}

/** Lee y parsea el archivo FASTA */
unordered_map<string, string> parse_fasta(const string &filepath) {
    cout << "Leyendo FASTA: " << filepath << endl;
    unordered_map<string, string> sequences;
    ifstream file(filepath);
    string line, current_id, current_seq;

    while (getline(file, line)) {
        if (line.empty()) continue;

        if (line[0] == '>') {
            if (!current_id.empty()) {
                sequences[current_id] = current_seq;
            }
            current_id = line.substr(1);
            current_seq.clear();
        } else {
            current_seq += line;
        }
    }

    if (!current_id.empty()) {
        sequences[current_id] = current_seq;
    }

    cout << "  Secuencias cargadas: " << sequences.size() << endl;
    return sequences;
}

// =============================================================================
// FUNCIONES DE LIMPIEZA
// =============================================================================

/** Elimina secuencias duplicadas basándose en hash del contenido */
void remove_duplicates(vector<Record> &records,
                       const unordered_map<string, string> &sequences) {
    cout << "\n=== PASO 2: Eliminando duplicados ===" << endl;

    unordered_set<size_t> seen_hashes;
    int no_match = 0, duplicates = 0;

    for (auto &r : records) {
        auto it = sequences.find(r.seq_id);
        if (it == sequences.end()) {
            r.valid = false;
            no_match++;
            continue;
        }

        r.seq_hash = fast_hash(it->second);

        if (seen_hashes.count(r.seq_hash)) {
            r.valid = false;
            duplicates++;
        } else {
            seen_hashes.insert(r.seq_hash);
        }
    }

    cout << "  Sin match en FASTA: " << no_match << endl;
    cout << "  Duplicados eliminados: " << duplicates << endl;

    int valid_count = count_if(records.begin(), records.end(),
                               [](const Record &r) { return r.valid; });
    cout << "  Secuencias unicas: " << valid_count << endl;
}

/**
 * Filtra outliers para UN nivel taxonómico.
 *
 * Para cada grupo dentro del nivel:
 *   1. Calcula media y desviación estándar de longitud
 *   2. Mantiene solo secuencias en rango [media - std/2, media + std/2]
 *   3. Grupos con n < 2 se incluyen completos
 *
 * @param records Vector de todos los registros
 * @param level Nivel taxonómico a filtrar
 * @return Vector de índices de registros válidos
 */
vector<size_t> filter_outliers_for_level(const vector<Record> &records,
                                         const string &level) {
    vector<size_t> valid_indices;
    unordered_map<string, vector<pair<size_t, int>>> groups;

    // Agrupar por nivel
    for (size_t i = 0; i < records.size(); i++) {
        const auto &r = records[i];
        if (!r.valid) continue;

        string group_name = get_tax_level(r, level);
        if (!group_name.empty()) {
            groups[group_name].push_back({i, r.length});
        }
    }

    // Filtrar por grupo
    for (const auto &[group_name, members] : groups) {
        // Grupos pequeños: incluir todos
        if (members.size() < 2) {
            for (const auto &[idx, len] : members) {
                valid_indices.push_back(idx);
            }
            continue;
        }

        // Calcular estadísticas
        double sum = 0;
        for (const auto &[idx, len] : members) sum += len;
        double mean = sum / members.size();

        double sq_sum = 0;
        for (const auto &[idx, len] : members)
            sq_sum += (len - mean) * (len - mean);
        double std_dev = sqrt(sq_sum / members.size());

        // Si std = 0, todos iguales
        if (std_dev == 0) {
            for (const auto &[idx, len] : members) {
                valid_indices.push_back(idx);
            }
            continue;
        }

        // Filtrar por rango
        double lower = mean - std_dev / 2;
        double upper = mean + std_dev / 2;

        for (const auto &[idx, len] : members) {
            if (len >= lower && len <= upper) {
                valid_indices.push_back(idx);
            }
        }
    }

    return valid_indices;
}

// =============================================================================
// FUNCIONES DE EXPORTACIÓN
// =============================================================================

/** Exporta dataset filtrado a CSV y FASTA */
void export_dataset(const vector<Record> &records,
                    const vector<size_t> &valid_indices,
                    const unordered_map<string, string> &sequences,
                    const string &level_name) {

    string csv_path = "data/data_clean_" + level_name + ".csv";
    string fasta_path = "data/data_clean_" + level_name + ".fasta";

    // CSV
    ofstream csv(csv_path);
    csv << "seq_id,length,kingdom,phylum,class,order,family,genus,species\n";

    for (size_t idx : valid_indices) {
        const auto &r = records[idx];
        csv << r.seq_id << "," << r.length << "," << r.kingdom << ","
            << r.phylum << "," << r.class_ << "," << r.order << ","
            << r.family << "," << r.genus << "," << r.species << "\n";
    }
    csv.close();

    // FASTA
    ofstream fasta(fasta_path);
    for (size_t idx : valid_indices) {
        const auto &r = records[idx];
        auto it = sequences.find(r.seq_id);
        if (it == sequences.end()) continue;

        fasta << ">" << r.seq_id << "\n";
        const string &seq = it->second;
        for (size_t i = 0; i < seq.size(); i += 60) {
            fasta << seq.substr(i, 60) << "\n";
        }
    }
    fasta.close();

    cout << "  " << setw(8) << left << level_name << ": " << setw(8) << right
         << valid_indices.size() << " secuencias -> " << csv_path << endl;
}

/** Imprime estadísticas de un nivel */
void print_level_stats(const vector<Record> &records,
                       const vector<size_t> &valid_indices,
                       const string &level_name) {
    if (valid_indices.empty()) return;

    double sum = 0;
    int min_len = INT_MAX, max_len = 0;

    for (size_t idx : valid_indices) {
        int len = records[idx].length;
        sum += len;
        min_len = min(min_len, len);
        max_len = max(max_len, len);
    }

    double mean = sum / valid_indices.size();

    double sq_sum = 0;
    for (size_t idx : valid_indices) {
        int len = records[idx].length;
        sq_sum += (len - mean) * (len - mean);
    }
    double std_dev = sqrt(sq_sum / valid_indices.size());

    cout << "    Stats: mean=" << fixed << setprecision(1) << mean
         << ", std=" << std_dev << ", range=[" << min_len << "-" << max_len
         << "]" << endl;
}

// =============================================================================
// MAIN
// =============================================================================

int main() {
    cout << "============================================================\n"
         << "DATA CLEANING PIPELINE - MIDORI2 COI (C++)\n"
         << "Genera 6 datasets independientes por nivel taxonomico\n"
         << "============================================================\n";

    // Paso 1: Cargar datos
    cout << "\n=== PASO 1: Cargando datos ===" << endl;
    auto records = parse_taxon(TAXON_FILE);
    auto sequences = parse_fasta(FASTA_FILE);

    // Paso 2: Eliminar duplicados
    remove_duplicates(records, sequences);

    int unique_count = count_if(records.begin(), records.end(),
                                [](const Record &r) { return r.valid; });

    // Paso 3: Generar dataset para cada nivel
    cout << "\n=== PASO 3: Filtrando y exportando por nivel ===" << endl;
    cout << "  Criterio: media +/- (std/2) dentro de cada grupo\n" << endl;

    for (const auto &tax_level : TAX_LEVELS) {
        auto valid_indices = filter_outliers_for_level(records, tax_level.name);
        export_dataset(records, valid_indices, sequences, tax_level.file_suffix);
        print_level_stats(records, valid_indices, tax_level.file_suffix);

        int removed = unique_count - valid_indices.size();
        cout << "    Removidos: " << removed << " (" << fixed << setprecision(1)
             << (100.0 * removed / unique_count) << "%)\n"
             << endl;
    }

    cout << "============================================================\n"
         << "Pipeline completado! Archivos generados:\n";
    for (const auto &tax_level : TAX_LEVELS) {
        cout << "  - data/data_clean_" << tax_level.file_suffix << ".csv\n"
             << "  - data/data_clean_" << tax_level.file_suffix << ".fasta\n";
    }
    cout << "============================================================" << endl;

    return 0;
}
