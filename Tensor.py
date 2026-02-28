import numpy as np

class Tensor(object):
    def __init__(self, data, creators=None, creation_op=None, autograd=False, id=None):
        self.data=np.array(data)
        self.creation_op=creation_op
        self.creators=creators
        self.grad=None
        self.autograd=autograd
        self.children={}
        self.index_select_indices = None
        self.softmax_output = None
        self.target_dist = None
        if id is None: id=np.random.randint(0, 100000)
        self.id=id

        if creators is not None:
            for c in creators:
                if(self.id not in c.children):
                    c.children[self.id]=1
                else:
                    c.children[self.id]+=1


    def all_children_grads_accounted_for(self, grad=None, grad_origin=None):
        for id,cnt in self.children.items():
            if cnt!=0: return False
        return True


    def backward(self, grad=None, grad_origin=None):
        if self.autograd:
            if grad is None:
                grad = np.ones_like(self.data)

            if isinstance(grad, Tensor):
                grad = grad.data

            if grad_origin is not None:
                if self.children[grad_origin.id]==0:
                    raise Exception("cannot backprop more than once")
                else:
                    self.children[grad_origin.id]-=1

            if self.grad is None:
                self.grad=grad
            else:
                self.grad+=grad

            if self.creators is not None and (self.all_children_grads_accounted_for() or grad_origin is None):
                if self.creation_op == "add":
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad, self)

                if self.creation_op == "neg":
                    self.creators[0].backward(self.grad * -1, self)

                if self.creation_op == "sub":
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad * -1, self)

                if self.creation_op == "mul":
                    self.creators[0].backward(self.grad * self.creators[1].data, self)
                    self.creators[1].backward(self.grad * self.creators[0].data, self)

                if self.creation_op == "mm":
                    act = self.creators[0]
                    weights = self.creators[1]
                    new_grad_act = self.grad.dot(weights.data.T)
                    act.backward(new_grad_act, self)

                    new_grad_weights = act.data.T.dot(self.grad)
                    weights.backward(new_grad_weights, self)

                if self.creation_op == "transpose":
                    self.creators[0].backward(self.grad.transpose(), self)

                if self.creation_op and "sum" in self.creation_op:
                    dim = int(self.creation_op.split("_")[1])
                    ds = self.creators[0].data.shape[dim]
                    grad_expandido = Tensor(self.grad).expand(dim, ds).data
                    self.creators[0].backward(grad_expandido, self)

                if self.creation_op and "expand" in self.creation_op:
                    dim = int(self.creation_op.split("_")[1])
                    self.creators[0].backward(self.grad.sum(dim), self)

                if self.creation_op == "sigmoid":
                    ones = np.ones_like(self.data)
                    sigmoid_grad = self.data * (ones - self.data)
                    self.creators[0].backward(self.grad * sigmoid_grad, self)

                if self.creation_op == "tanh":
                    ones = np.ones_like(self.data)
                    tanh_grad = ones - (self.data**2)
                    self.creators[0].backward(self.grad * tanh_grad, self)

                if self.creation_op == "index_select":
                    if self.index_select_indices is not None and self.creators is not None:
                        new_grad = np.zeros_like(self.creators[0].data)

                        indices_ = self.index_select_indices.data.flatten()

                        grad_flat = self.grad.reshape(len(indices_), -1)

                        for i in range(len(indices_)):
                            new_grad[indices_[i]] += grad_flat[i]

                        self.creators[0].backward(new_grad, self)

                if self.creation_op == "cross_entropy":
                    if self.softmax_output is not None and self.target_dist is not None:
                        dx = self.softmax_output - self.target_dist
                        dx /= len(self.target_dist)
                        self.creators[0].backward(dx, self)




    def __add__(self, other):
        if self.autograd or other.autograd:
            return Tensor(self.data+other.data, creators=[self, other], creation_op="add", autograd=True)
        return Tensor(self.data + other.data)

    def __neg__(self):
        if self.autograd:
            return Tensor(self.data * -1, creators=[self], creation_op="neg", autograd=True)
        return Tensor(self.data * -1)

    def __sub__(self,other):
        if self.autograd or other.autograd:
            return Tensor(self.data- other.data, autograd=True, creators=[self,other], creation_op="sub")
        return Tensor(self.data - other.data)

    def __mul__(self,other):
        if self.autograd or other.autograd:
            return Tensor(self.data * other.data, autograd=True, creators=[self,other], creation_op="mul")
        return Tensor(self.data * other.data)

    def sum(self, dim):
        if self.autograd:
            return Tensor(self.data.sum(dim), autograd=True, creators=[self], creation_op=f"sum_{dim}")
        return Tensor(self.data.sum(dim))

    def expand(self, dim, copies):
        trans_cmd=list(range(0, len(self.data.shape)))
        trans_cmd.insert(dim, len(self.data.shape))
        new_shape=list(self.data.shape)+[copies]
        new_data=self.data.repeat(copies).reshape(new_shape)
        new_data=new_data.transpose(trans_cmd)

        if self.autograd:
            return Tensor(new_data, autograd=True, creators=[self], creation_op=f"expand_{dim}")
        return Tensor(new_data)

    def transpose(self):
        if self.autograd:
            return Tensor(self.data.transpose(), autograd=True, creators=[self], creation_op="transpose")
        return Tensor(self.data.transpose())

    def mm(self, x):
        if self.autograd or x.autograd:
            return Tensor(self.data.dot(x.data), autograd=True, creators=[self, x], creation_op="mm")
        return Tensor(self.data.dot(x.data))

    def sigmoid(self):
        if self.autograd:
            out = 1 / (1 + np.exp(-self.data))
            return Tensor(out, autograd=True, creators=[self], creation_op="sigmoid")
        return Tensor(1 / (1 + np.exp(-self.data)))

    def tanh(self):
        if self.autograd:
            return Tensor(np.tanh(self.data), autograd=True, creators=[self], creation_op="tanh")
        return Tensor(np.tanh(self.data))

    def index_select(self, indices):
        if self.autograd:
            new = Tensor(self.data[indices.data],
                         autograd=True,
                         creators=[self],
                         creation_op="index_select")
            new.index_select_indices = indices
            return new
        return Tensor(self.data[indices.data])

    def cross_entropy(self, target_indices):
        temp = np.exp(self.data - np.max(self.data, axis=-1, keepdims=True))
        softmax_output = temp / np.sum(temp, axis=len(self.data.shape)-1, keepdims=True)

        t = target_indices.data.flatten()
        p = softmax_output.reshape(len(t), -1)
        target_dist = np.eye(p.shape[1])[t]

        loss = -(np.log(p + 1e-15) * target_dist).sum(1).mean()

        if self.autograd:
            out = Tensor(loss,
                         autograd=True,
                         creators=[self],
                         creation_op="cross_entropy")
            out.softmax_output = softmax_output
            out.target_dist = target_dist
            return out
        return Tensor(loss)


    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())



