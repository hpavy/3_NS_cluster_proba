# Avec Z et le choix des points avec une certaine proba
from deepxrte.geometry import Rectangle
import torch
import torch.nn as nn
import torch.optim as optim
from model import PINNs
from utils import read_csv, write_csv
from train import train
from pathlib import Path
import time
import pandas as pd
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

time_start = time.time()

############# LES VARIABLES ################

folder_result = '4_supervised'  # le nom du dossier de résultat

random_seed_train = np.random.randint(1, 100000)  # la seed de train, la changer pour avoir de nouveau résultats
random_seed_test = 2002   # la seed de test, toujours garder la même pour pouvoir comparer



##### Le modèle de résolution de l'équation de la chaleur
nb_itt = 20000     # le nb d'epoch
resample_rate = 100  # le taux de resampling
display = 500       # le taux d'affichage
poids = [1, 0]   # les poids pour la loss

n_data = 10000         # le nb de points initiaux
n_pde = 1          # le nb de points pour la pde

n_data_test = 5000
n_pde_test = 5000

Re = 100

lr = 1e-4

x_proba_max = 0.3
x_proba_min = 0.03
y_proba_max = 0.15
y_proba_min = 0.05

proba = 0.1

##### Le code ###############################
###############################################

# La data
df = pd.read_csv('data.csv')


# On adimensionne la data
df_modified = df[(df['Points:0'] >= -0.07) & (df['Points:1'] >= -0.1) & (df['Points:1'] <= 0.1)]
x, y, t = (np.array(df_modified['Points:0']), np.array(df_modified['Points:1']),
           np.array(df_modified['Time']))
u, v, p = (np.array(df_modified['Velocity:0']), np.array(df_modified['Velocity:1']),
           np.array(df_modified['Pressure']))
x = x-x.min()
y = y-y.min()

x_norm = (x-x.mean())/x.std()
y_norm = (y-y.mean())/y.std()
t_norm = (t-t.mean())/t.std()
p_norm = (p-p.mean())/p.std()
u_norm = (u-u.mean())/u.std()
v_norm = (v-v.mean())/v.std()


x_proba_norm_max = (x_proba_max-x.mean())/x.std()
x_proba_norm_min = (x_proba_min-x.mean())/x.std()
y_proba_norm_max = (y_proba_max-y.mean())/y.std()
y_proba_norm_min = (y_proba_min-y.mean())/y.std()


X = np.array([x_norm, y_norm, t_norm], dtype=np.float32).T
U = np.array([u_norm, v_norm, p_norm], dtype=np.float32).T

t_norm_min = t_norm.min()
t_norm_max = t_norm.max()
t_max = t.max()

x_norm_max = x_norm.max()
y_norm_max = y_norm.max()


# On regarde si le dossier existe
dossier = Path(folder_result)
dossier.mkdir(parents=True, exist_ok=True)

torch.manual_seed(random_seed_train)

rectangle_proba = Rectangle(x_max=x_proba_norm_max, y_max=y_proba_norm_max,
                            t_min=t_norm_min, t_max=t_norm_max)    # le domaine de plus haute proba

rectangle = Rectangle(x_max=x_norm_max, y_max=y_norm_max,
                      t_min=t_norm_min, t_max=t_norm_max)    # le domaine de résolution


# les points initiaux du train
# Les points de pde


### Pour train
# les points pour la pde
points_pde_proba = rectangle_proba.generate_random(int(n_pde*proba)).to(device)
points_pde_non_proba = rectangle.generate_random(n_pde-int(n_pde*proba)).to(device)
points_pde = torch.cat((points_pde_proba, points_pde_non_proba), 0)


# On prend des points de data uniquement dans notre rectangle de proba
np.random.seed(random_seed_train)
masque = ((X[:, 0] > x_proba_min) & (X[:, 0] < x_proba_max) &
          (X[:, 1] > y_proba_min) & (X[:, 1] < y_proba_max))
points_data_train_proba = np.random.choice(len(X[masque]), int(n_data*proba), replace=False)
inputs_train_data_proba = torch.from_numpy(X[masque][points_data_train_proba]).requires_grad_().to(device)
outputs_train_data_proba = torch.from_numpy(U[masque][points_data_train_proba]).requires_grad_().to(device)
points_data_train_non_proba = np.random.choice(len(X), n_data-int(n_data*proba), replace=False)
inputs_train_data_non_proba = torch.from_numpy(X[points_data_train_non_proba]).requires_grad_().to(device)
outputs_train_data_non_proba = torch.from_numpy(U[points_data_train_non_proba]).requires_grad_().to(device)
inputs_train_data = torch.cat((inputs_train_data_proba, inputs_train_data_non_proba), 0)
outputs_train_data = torch.cat((outputs_train_data_proba, outputs_train_data_non_proba), 0)



### Pour test
torch.manual_seed(random_seed_test)
np.random.seed(random_seed_test)
X_test_pde = rectangle.generate_random(n_pde_test).to(device)
points_coloc_test = np.random.choice(len(X), n_data_test, replace=False)
X_test_data = torch.from_numpy(X[points_coloc_test]).requires_grad_().to(device)
U_test_data = torch.from_numpy(U[points_coloc_test]).requires_grad_().to(device)



# Initialiser le modèle
model = PINNs().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
loss = nn.MSELoss()

# On plot les print dans un fichier texte 
with open(folder_result+'/print.txt', 'a') as f:
    # On regarde si notre modèle n'existe pas déjà 
    if Path(folder_result+'/model_weights.pth').exists():
        # Charger l'état du modèle et de l'optimiseur
        checkpoint = torch.load(folder_result+'/model_weights.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("\nModèle chargé\n", file=f)
        print("\nModèle chargé\n")
        train_loss = read_csv(folder_result+'/train_loss.csv')['0'].to_list()
        test_loss = read_csv(folder_result+'/test_loss.csv')['0'].to_list()
        print("\nLoss chargée\n", file=f)
        print("\nLoss chargée\n")

    else:
        print('Nouveau modèle\n', file=f)
        print('Nouveau modèle\n')
        train_loss = []
        test_loss = []


    ######## On entraine le modèle
    ###############################################
    train(nb_itt=nb_itt, train_loss=train_loss, test_loss=test_loss, resample_rate=resample_rate,
          display=display, poids=poids, inputs_train_data=inputs_train_data,
          outputs_train_data=outputs_train_data, points_pde=points_pde,
          model=model, loss=loss, optimizer=optimizer, X=X, U=U, n_pde=n_pde, X_test_pde=X_test_pde,
          X_test_data=X_test_data, U_test_data=U_test_data, n_data=n_data, rectangle=rectangle,
          device=device, Re=Re, time_start=time_start, f=f,
          u_mean=u.mean(), v_mean=v.mean(), x_std=x.std(), y_std=y.std(),
          t_std=t.std(), u_std=u.std(), v_std=v.std(), p_std=p.std(), masque=masque,
          proba=proba, rectangle_proba=rectangle_proba)

####### On save le model et les losses

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, folder_result+'/model_weights.pth')
write_csv(train_loss, folder_result, file_name='/train_loss.csv')
write_csv(test_loss, folder_result, file_name='/test_loss.csv')



dossier_end = Path(folder_result + f"/epoch{len(train_loss)}")
dossier_end.mkdir(parents=True, exist_ok=True)
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, folder_result+f"/epoch{len(train_loss)}"+'/model_weights.pth')
write_csv(train_loss, folder_result + f"/epoch{len(train_loss)}", file_name='/train_loss.csv')
write_csv(test_loss, folder_result + f"/epoch{len(train_loss)}", file_name='/test_loss.csv')