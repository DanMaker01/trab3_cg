import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import glm
import math
from PIL import Image
from objeto import Objeto

class Luz(Objeto):
    def __init__(self, posicao=(0.0, 0.0, 0.0), ka=1.0, kd=1.0, ks=1.0, ns=100.0):
        super().__init__()  # Inicializa os atributos da classe base
        self.posicao = glm.vec3(posicao)
        self.ka = ka  # Coeficiente de reflexão ambiente
        self.kd = kd  # Coeficiente de reflexão difusa
        self.ks = ks  # Coeficiente de reflexão especular
        self.ns = ns  # Expoente especular

    def atualiza_posicao(self, x, y, z):
        """Atualiza a posição da luz."""
        self.posicao = glm.vec3(x, y, z)

    def envia_parametros_para_gpu(self, program):
        """Envia os parâmetros da luz para a GPU."""
        loc_ka = glGetUniformLocation(program, "ka")
        glUniform1f(loc_ka, self.ka)

        loc_kd = glGetUniformLocation(program, "kd")
        glUniform1f(loc_kd, self.kd)

        loc_ks = glGetUniformLocation(program, "ks")
        glUniform1f(loc_ks, self.ks)

        loc_ns = glGetUniformLocation(program, "ns")
        glUniform1f(loc_ns, self.ns)

        loc_light_pos = glGetUniformLocation(program, "lightPos")
        glUniform3f(loc_light_pos, self.posicao.x, self.posicao.y, self.posicao.z)

    def desenha(self, program, model_matrix=glm.mat4(1.0)):
        """
        Redefine o método desenha.
        Uma luz geralmente não precisa ser desenhada como um objeto visual.
        Se necessário, pode ser representada como uma esfera ou ponto no espaço.
        """
        super().desenha(program, model_matrix)  # Mantém o comportamento original se aplicável.
