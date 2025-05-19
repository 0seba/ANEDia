if __name__ == "__main__":
    import argparse 
    import coremltools
    import shutil

    
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)

    args = parser.parse_args()
    mlmodel = coremltools.models.MLModel(args.model_path, compute_units=coremltools.ComputeUnit.CPU_AND_NE)
    compiled_path = mlmodel.get_compiled_model_path()
    shutil.copytree(compiled_path, args.model_path.strip(".mlpackage") + ".mlmodelc")
    
    