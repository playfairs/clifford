import argparse
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

from clifford import (
    Network,
    Dense,
    ReLU,
    Sigmoid,
    Softmax,
    MeanSquaredError,
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
    SGD,
    Adam,
    Trainer,
    Dataset,
    ModelPersistence,
    ModelRegistry,
    SearchEngine,
    ConfigManager,
    Accuracy,
    CliffordError,
    ModelNotFoundError,
    TrainingError,
)
from clifford.utils import get_asset_path


def train_command(args):
    try:
        config_manager = ConfigManager()
        config = config_manager.load()
        
        registry = ModelRegistry()
        persistence = ModelPersistence()
        
        if args.dataset:
            dataset = Dataset.load(Path(args.dataset))
        else:
            from clifford import generate_xor_dataset
            dataset = generate_xor_dataset(samples=1000)
            dataset.normalize()
        
        network = Network(name=args.model_name)
        
        architecture = args.architecture.split(",")
        for layer_spec in architecture:
            parts = layer_spec.split(":")
            units = int(parts[0])
            activation = parts[1] if len(parts) > 1 else "relu"
            
            if activation == "relu":
                act = ReLU()
            elif activation == "sigmoid":
                act = Sigmoid()
            elif activation == "softmax":
                act = Softmax()
            else:
                act = ReLU()
            
            network.add(Dense(units=units, activation=act))
        
        if args.loss == "mse":
            loss = MeanSquaredError()
        elif args.loss == "bce":
            loss = BinaryCrossEntropy()
        elif args.loss == "cce":
            loss = CategoricalCrossEntropy()
        else:
            loss = MeanSquaredError()
        
        if args.optimizer == "sgd":
            optimizer = SGD(learning_rate=args.learning_rate)
        elif args.optimizer == "adam":
            optimizer = Adam(learning_rate=args.learning_rate)
        else:
            optimizer = SGD(learning_rate=args.learning_rate)
        
        network.compile(loss=loss, optimizer=optimizer)
        
        trainer = Trainer(network, config=config.training)
        trainer.config.epochs = args.epochs
        trainer.config.batch_size = args.batch_size
        trainer.config.verbose = args.verbose
        
        print(f"Training model: {args.model_name}")
        print(network.summary())
        
        result = trainer.train(dataset.X, dataset.y)
        
        model_id = registry.register_model(
            name=args.model_name,
            architecture=network.get_architecture(),
            hyperparameters={"epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate}
        )
        
        run_id = registry.register_training_run(model_id, trainer.config.__dict__)
        registry.update_training_run(run_id, end_time=result["end_time"], status="completed", final_loss=result["final_loss"])
        
        if args.save:
            persistence.save_model(network, args.model_name)
            print(f"Model saved: {args.model_name}")
        
        print(f"Training completed. Final loss: {result['final_loss']:.6f}")
        
    except CliffordError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def evaluate_command(args):
    try:
        persistence = ModelPersistence()
        network = persistence.load_model(args.model_name)
        
        if args.dataset:
            dataset = Dataset.load(Path(args.dataset))
        else:
            from clifford import generate_xor_dataset
            dataset = generate_xor_dataset(samples=100)
            dataset.normalize()
        
        results = network.evaluate(dataset.X, dataset.y)
        
        accuracy = Accuracy()
        y_pred = network.predict(dataset.X)
        acc = accuracy.compute(dataset.y, y_pred)
        
        print(f"Evaluation results for {args.model_name}:")
        print(f"  Loss: {results['loss']:.6f}")
        print(f"  Accuracy: {acc:.6f}")
        
    except ModelNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except CliffordError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def predict_command(args):
    try:
        persistence = ModelPersistence()
        network = persistence.load_model(args.model_name)
        
        input_data = np.array([float(x) for x in args.input.split(",")], dtype=np.float64).reshape(1, -1)
        prediction = network.predict(input_data)
        
        print(f"Prediction: {prediction.flatten()}")
        
    except ModelNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except CliffordError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def save_command(args):
    try:
        persistence = ModelPersistence()
        
        if args.weights_only:
            persistence.save_weights(network, args.model_name)
            print(f"Weights saved: {args.model_name}")
        else:
            print("Error: Model not loaded. Use train command first.", file=sys.stderr)
            sys.exit(1)
        
    except CliffordError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def load_command(args):
    try:
        persistence = ModelPersistence()
        network = persistence.load_model(args.model_name)
        print(f"Model loaded: {args.model_name}")
        print(network.summary())
        
    except ModelNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except CliffordError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def list_models_command(args):
    try:
        registry = ModelRegistry()
        models = registry.list_models()
        
        if not models:
            print("No models found.")
            return
        
        print("Registered models:")
        for model in models:
            print(f"  - {model.name} (created: {model.created_at})")
        
    except CliffordError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def list_runs_command(args):
    try:
        registry = ModelRegistry()
        runs = registry.list_training_runs()
        
        if not runs:
            print("No training runs found.")
            return
        
        print("Training runs:")
        for run in runs:
            print(f"  - {run.id} (status: {run.status}, loss: {run.final_loss})")
        
    except CliffordError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def search_command(args):
    try:
        search_engine = SearchEngine()
        
        if args.type == "models":
            results = search_engine.search_models(name=args.query)
            print(f"Found {len(results)} models matching '{args.query}':")
            for model in results:
                print(f"  - {model.name}")
        elif args.type == "runs":
            results = search_engine.search_training_runs(status=args.query)
            print(f"Found {len(results)} runs with status '{args.query}':")
            for run in results:
                print(f"  - {run.id}")
        elif args.type == "datasets":
            results = search_engine.search_datasets(name=args.query)
            print(f"Found {len(results)} datasets matching '{args.query}':")
            for dataset in results:
                print(f"  - {dataset.name}")
        else:
            print("Error: Invalid search type. Use 'models', 'runs', or 'datasets'.", file=sys.stderr)
            sys.exit(1)
        
    except CliffordError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Clifford Neural Network Framework CLI")
    parser.add_argument("--version", action="version", version="Clifford 0.1.0")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    train_parser = subparsers.add_parser("train", help="Train a neural network")
    train_parser.add_argument("--model-name", required=True, help="Name for the model")
    train_parser.add_argument("--architecture", required=True, help="Layer architecture (e.g., '64:relu,32:sigmoid,1:sigmoid')")
    train_parser.add_argument("--loss", default="mse", choices=["mse", "bce", "cce"], help="Loss function")
    train_parser.add_argument("--optimizer", default="sgd", choices=["sgd", "adam"], help="Optimizer")
    train_parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    train_parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate")
    train_parser.add_argument("--dataset", help="Path to dataset file")
    train_parser.add_argument("--save", action="store_true", help="Save the trained model")
    train_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    train_parser.set_defaults(func=train_command)
    
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    evaluate_parser.add_argument("--model-name", required=True, help="Name of the model to evaluate")
    evaluate_parser.add_argument("--dataset", help="Path to dataset file")
    evaluate_parser.set_defaults(func=evaluate_command)
    
    predict_parser = subparsers.add_parser("predict", help="Make predictions with a trained model")
    predict_parser.add_argument("--model-name", required=True, help="Name of the model")
    predict_parser.add_argument("--input", required=True, help="Input values (comma-separated)")
    predict_parser.set_defaults(func=predict_command)
    
    save_parser = subparsers.add_parser("save", help="Save a model")
    save_parser.add_argument("--model-name", required=True, help="Name of the model")
    save_parser.add_argument("--weights-only", action="store_true", help="Save only weights")
    save_parser.set_defaults(func=save_command)
    
    load_parser = subparsers.add_parser("load", help="Load a model")
    load_parser.add_argument("--model-name", required=True, help="Name of the model")
    load_parser.set_defaults(func=load_command)
    
    list_models_parser = subparsers.add_parser("list-models", help="List all registered models")
    list_models_parser.set_defaults(func=list_models_command)
    
    list_runs_parser = subparsers.add_parser("list-runs", help="List all training runs")
    list_runs_parser.set_defaults(func=list_runs_command)
    
    search_parser = subparsers.add_parser("search", help="Search models, runs, or datasets")
    search_parser.add_argument("--type", required=True, choices=["models", "runs", "datasets"], help="Type to search")
    search_parser.add_argument("--query", required=True, help="Search query")
    search_parser.set_defaults(func=search_command)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()
