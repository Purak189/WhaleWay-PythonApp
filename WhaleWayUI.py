import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk

# Leer el archivo de tiendas y sus nodos
def leer_tiendas(archivo):
    tiendas = {}
    with open(archivo, 'r') as f:
        for linea in f:
            nodo, nombre = linea.strip().split(', ')
            tiendas[nombre] = int(nodo)
    return tiendas

# Función para agregar un pedido a la lista
def agregar_pedido():
    tienda_seleccionada = tienda_var.get()
    cantidad_productos = int(cantidad_var.get())
    if tienda_seleccionada and cantidad_productos > 0:
        pedidos.append((tienda_seleccionada, cantidad_productos))
        actualizar_lista_pedidos()
        top.destroy()
    else:
        messagebox.showwarning("Entrada inválida", "Por favor seleccione una tienda y una cantidad válida.")

# Función para abrir la ventana emergente para agregar pedidos
def abrir_ventana_agregar():
    global top
    top = tk.Toplevel(root)
    top.title("Agregar Pedido")
    top.geometry("300x200")

    tk.Label(top, text="Tiendas:", bg="#E0F7FA").pack(pady=5)
    tienda_menu = ttk.Combobox(top, textvariable=tienda_var, values=list(tiendas.keys()))
    tienda_menu.pack(pady=5)

    tk.Label(top, text="Cantidad de productos:", bg="#E0F7FA").pack(pady=5)
    tk.Entry(top, textvariable=cantidad_var).pack(pady=5)

    tk.Button(top, text="Agregar tienda", command=agregar_pedido, bg="#0288D1", fg="#FFFFFF", font=("Arial", 12)).pack(pady=10)

# Función para actualizar la lista de pedidos en la ventana principal
def actualizar_lista_pedidos():
    for row in tree.get_children():
        tree.delete(row)
    for i, (tienda, cantidad) in enumerate(pedidos, 1):
        tree.insert("", "end", text=str(i), values=(tienda, f"{cantidad} productos"))

# Inicializar la ventana principal
root = tk.Tk()
root.title("WhaleWay")
root.geometry("640x540")

# Personalizar colores y estilos
root.configure(bg="#E0F7FA")  # Fondo de la app
style = ttk.Style()
style.configure("TButton", background="#0288D1", foreground="#FFFFFF", font=("Arial", 12))
style.configure("Treeview", font=("Arial", 12), rowheight=25)
style.configure("Treeview.Heading", font=("Arial", 12, "bold"))

# Título de la aplicación
tk.Label(root, text="WhaleWay", font=("Arial", 24), bg="#E0F7FA").grid(row=0, column=1, pady=10)

# Cargar y mostrar la imagen del logo
logo = Image.open("logo.png")  # Reemplaza con la ruta a tu imagen
logo = logo.resize((100, 100), Image.LANCZOS)
logo = ImageTk.PhotoImage(logo)
tk.Label(root, image=logo, bg="#E0F7FA").grid(row=0, column=0, padx=10, pady=10)


# Botón para agregar pedido
tk.Button(root, text="Agregar pedido", command=abrir_ventana_agregar, bg="#0288D1", fg="#FFFFFF", font=("Arial", 12)).grid(row=1, column=0, columnspan=2, pady=10)

# Lista de pedidos
tree = ttk.Treeview(root, columns=("Tienda", "Pedido"))
tree.heading("#0", text="ID")
tree.heading("Tienda", text="Tienda")
tree.heading("Pedido", text="Pedido")
tree.grid(row=2, column=0, columnspan=2, padx=20, pady=10)

# Botón para mostrar recorrido
tk.Button(root, text="Mostrar recorrido", command=lambda: print("Mostrando recorrido..."), bg="#0288D1", fg="#FFFFFF", font=("Arial", 12)).grid(row=3, column=0, columnspan=2, pady=10)

# Variables para los widgets
tienda_var = tk.StringVar()
cantidad_var = tk.IntVar()
pedidos = []
tiendas = leer_tiendas("Nodos_Tiendas_ids.txt")

# Iniciar el loop de la aplicación
root.mainloop()
