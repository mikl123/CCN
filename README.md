# CCN

Code and data for the paper "[Multi-Label Classification Neural Networks with Hard Logical Constraints](https://dl.acm.org/doi/pdf/10.1613/jair.1.12850)"

## Configure python environment

Step 1: Create the virtual environment:
```bash
python3 -m venv ccnenv
```

Step 2: Activate the virtual environment:

- On Windows: `ccnenv\Scripts\activate`
- On macOS/Linux `source ccnenv/bin/activate`

Step 3: Install the required packages:
```bash
pip install -r requirements.txt
```

## Train CCN

In order to train the network use the file ```train.py```. An example on how to run it for the dataset ```emotions``` is the following:
```
python train.py --dataset "emotions" --num_classes 6 --seed "$seed" --split "$split" --device "$device" --lr "$lr" --dropout "$dropout" --hidden_dim "$hidden_dim"  --num_layers "$num_layers" --weight_decay "$weight_decay" --non_lin "$non_lin" --batch_size "$batch_size" 
```
```train.py``` saves a pickle file for each execution in the ```hyp``` folder. Each pickle file stores the value of the hyperparameters and of the validation loss.

The hyperparameter used for training are: --batch_size 4 --lr 0.0001 --dropout 0.8 --weight_decay 0.0001 . The patience of 20 is specified in the code and hidden dimention size is differen for each dataset and can be taken from the paper from the Table 13.

## Test CCN 

In order to test the network use the file ```test.py```. An example on how to run it for the dataset ```emotions``` is the following: 
```
python test.py --dataset emotions --seed "$seed" --device 1&
```
```test.py``` finds the combinations of hyperparameters that results in the best validation loss, train the relative model, and then writes the results in the ```results``` folder.

## Reference

```
@article{giunchiglia2021,
  author    = {Eleonora Giunchiglia and Thomas Lukasiewicz},
  title     = {Multi-Label Classification Neural Networks with Hard Logical Constraints},
  journal   = {Journal of Artificial Iintelligence Research (JAIR)},
  volume    = {72},
  year      = {2021}
}
```
