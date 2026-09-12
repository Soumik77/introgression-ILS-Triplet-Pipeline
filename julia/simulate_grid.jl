using CSV
using DataFrames
using PhyloCoalSimulations
using PhyloNetworks
using Random

function parse_args(args)
    opts = Dict{String,String}()
    flags = Set{String}()
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--overwrite"
            push!(flags, a)
            i += 1
        elseif startswith(a, "--") && i < length(args)
            opts[a] = args[i + 1]
            i += 2
        else
            error("Unknown or incomplete argument: $a")
        end
    end
    return opts, flags
end

function count_nonempty_lines(path)
    n = 0
    open(path, "r") do io
        for line in eachline(io)
            if !isempty(strip(line))
                n += 1
            end
        end
    end
    return n
end

opts, flags = parse_args(ARGS)
manifest_path = get(opts, "--manifest", "")
out_dir = get(opts, "--out-dir", "")
isempty(manifest_path) && error("--manifest is required")
isempty(out_dir) && error("--out-dir is required")

manifest = CSV.read(manifest_path, DataFrame)
start_row = parse(Int, get(opts, "--start-row", "1"))
end_row = parse(Int, get(opts, "--end-row", string(nrow(manifest))))
1 <= start_row <= end_row <= nrow(manifest) || error("Invalid row range")
mkpath(out_dir)

for row_index in start_row:end_row
    row = manifest[row_index, :]
    out_path = joinpath(out_dir, string(row.simulation_id) * ".tre")
    expected = Int(row.gene_tree_count)

    if isfile(out_path) && !("--overwrite" in flags)
        if count_nonempty_lines(out_path) == expected
            println("skip complete row=$row_index simulation=$(row.simulation_id)")
            continue
        end
    end

    Random.seed!(Int(row.seed))
    network = readnewick(String(row.network_newick))
    trees = simulatecoalescent(network, expected, 1)

    tmp_path = out_path * ".partial"
    writemultinewick(trees, tmp_path)
    count_nonempty_lines(tmp_path) == expected || error("Incomplete simulation: $(row.simulation_id)")
    mv(tmp_path, out_path; force=true)
    println("done row=$row_index simulation=$(row.simulation_id) trees=$expected")
end

