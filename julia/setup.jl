using Pkg

project_dir = @__DIR__
Pkg.activate(project_dir)
Pkg.add(["CSV", "DataFrames", "PhyloCoalSimulations", "PhyloNetworks"])
Pkg.instantiate()
println("Julia simulation environment is ready at $project_dir")

