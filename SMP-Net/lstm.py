
import torch
import math
import torch.nn as nn
class LSTM_1(torch.nn.Module):

    def __init__(self, input_length, hidden_length):
        super(LSTM_1, self).__init__()
        self.hidden_length = hidden_length
        self.input_length = input_length

        self.multivariate = nn.Linear(self.input_length,self.input_length,bias=True)

        # input gate components
        self.linear_input_W_x_task0 = nn.Linear(self.input_length+self.hidden_length, self.hidden_length, bias=True)
        self.linear_input_W_h_task0 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.linear_input_w_c_task0 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_input_task0 = nn.Sigmoid()

        # forget gate components
        self.linear_forget_W_x_task0 = nn.Linear(self.input_length+self.hidden_length, self.hidden_length, bias=True)
        self.linear_forget_W_h_task0 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.linear_forget_w_c_task0 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_forget_task0 = nn.Sigmoid()

        self.linear_cell_W_x_task0 = nn.Linear(self.input_length+self.hidden_length, self.hidden_length, bias=True)
        self.linear_cell_W_h_task0 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.tanh_cell_task0 = nn.Tanh()

        # out gate components
        self.linear_output_W_x_task0 = nn.Linear(self.input_length+self.hidden_length, self.hidden_length, bias=True)
        self.linear_output_W_h_task0 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.linear_output_w_c_task0 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_out_task0 = nn.Sigmoid()


        #=====================================================================================
        # input gate components
        self.linear_input_W_x_task1 = nn.Linear(self.input_length + self.hidden_length, self.hidden_length, bias=True)
        self.linear_input_W_h_task1 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.linear_input_w_c_task1 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_input_task1 = nn.Sigmoid()

        # forget gate components
        self.linear_forget_W_x_task1 = nn.Linear(self.input_length + self.hidden_length, self.hidden_length, bias=True)
        self.linear_forget_W_h_task1 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.linear_forget_w_c_task1 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_forget_task1 = nn.Sigmoid()

        # cell memory components
        self.linear_cell_W_x_task1 = nn.Linear(self.input_length + self.hidden_length, self.hidden_length, bias=True)
        self.linear_cell_W_h_task1 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.tanh_cell_task1 = nn.Tanh()

        # out gate components
        self.linear_output_W_x_task1 = nn.Linear(self.input_length + self.hidden_length, self.hidden_length, bias=True)
        self.linear_output_W_h_task1 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.linear_output_w_c_task1 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        # self.linear_output_W_t_task1 = nn.Linear(TIME_EMBED_SIZE, self.hidden_length, bias=False)
        self.sigmoid_out_task1 = nn.Sigmoid()

        # =====================================================================================
        # input gate components
        self.linear_input_W_x_task2 = nn.Linear(self.input_length + self.hidden_length, self.hidden_length, bias=True)
        self.linear_input_W_h_task2 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.linear_input_w_c_task2 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_input_task2 = nn.Sigmoid()

        # forget gate components
        self.linear_forget_W_x_task2 = nn.Linear(self.input_length + self.hidden_length, self.hidden_length, bias=True)
        self.linear_forget_W_h_task2 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.linear_forget_w_c_task2 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_forget_task2 = nn.Sigmoid()

        # cell memory components
        self.linear_cell_W_x_task2 = nn.Linear(self.input_length + self.hidden_length, self.hidden_length, bias=True)
        self.linear_cell_W_h_task2 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.tanh_cell_task2 = nn.Tanh()

        # out gate components
        self.linear_output_W_x_task2 = nn.Linear(self.input_length + self.hidden_length, self.hidden_length, bias=True)
        self.linear_output_W_h_task2 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.linear_output_w_c_task2 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        # self.linear_output_W_t_task2 = nn.Linear(TIME_EMBED_SIZE, self.hidden_length, bias=False)
        self.sigmoid_out_task2 = nn.Sigmoid()
        # =====================================================================================
        # input gate components
        self.linear_input_W_x_task3 = nn.Linear(self.input_length + self.hidden_length, self.hidden_length, bias=True)
        self.linear_input_W_h_task3 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.linear_input_w_c_task3 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_input_task3 = nn.Sigmoid()

        # forget gate components
        self.linear_forget_W_x_task3 = nn.Linear(self.input_length + self.hidden_length, self.hidden_length, bias=True)
        self.linear_forget_W_h_task3 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.linear_forget_w_c_task3 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_forget_task3 = nn.Sigmoid()

        # cell memory components
        self.linear_cell_W_x_task3 = nn.Linear(self.input_length + self.hidden_length, self.hidden_length, bias=True)
        self.linear_cell_W_h_task3 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.tanh_cell_task3 = nn.Tanh()

        # out gate components
        self.linear_output_W_x_task3 = nn.Linear(self.input_length + self.hidden_length, self.hidden_length, bias=True)
        self.linear_output_W_h_task3 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.linear_output_w_c_task3 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        # self.linear_output_W_t_task3 = nn.Linear(TIME_EMBED_SIZE, self.hidden_length, bias=False)
        self.sigmoid_out_task3 = nn.Sigmoid()

        # input gate shared components
        self.linear_input_W_x_shared = nn.Linear(self.input_length+ self.hidden_length , self.hidden_length, bias=True)
        self.linear_input_W_h_shared = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.linear_input_w_c_shared = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_input_shared = nn.Sigmoid()

        # forget gate shared components
        self.linear_forget_W_x_shared = nn.Linear(self.input_length+ self.hidden_length , self.hidden_length, bias=True)
        self.linear_forget_W_h_shared = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.linear_forget_w_c_shared = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_forget_shared = nn.Sigmoid()

        # cell memory shared components
        self.linear_cell_W_x_shared = nn.Linear(self.input_length+ self.hidden_length , self.hidden_length, bias=True)
        self.linear_cell_W_h_shared = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.tanh_cell_shared = nn.Tanh()

        # out gate shared components
        self.linear_output_W_x_shared = nn.Linear(self.input_length+ self.hidden_length , self.hidden_length, bias=True)
        self.linear_output_W_h_shared = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.linear_output_w_c_shared = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        # self.linear_output_W_t_shared = nn.Linear(TIME_EMBED_SIZE, self.hidden_length, bias=False)
        self.sigmoid_out_shared = nn.Sigmoid()

        # feature attention
        self.linear_feature_attention_W_x = nn.Linear(self.input_length, self.input_length, bias=True)
        self.sigmoid_feature_attention = nn.Sigmoid()
        # shared_layer attention
        self.linear_shared_layer_attention_W_x = nn.Linear(self.hidden_length, self.hidden_length, bias=True)
        self.sigmoid_shared_layer_attention = nn.Sigmoid()
    def input_gate(self, x, h, c_prev,shared,taskname):
        if shared:
            x_temp = self.linear_input_W_x_shared(x)
            h_temp = self.linear_input_W_h_shared(h)
            c_temp = self.linear_input_w_c_shared(c_prev)
            i = self.sigmoid_input_shared(x_temp + h_temp)
            return i
        else:
            if taskname==0:
                x_temp = self.linear_input_W_x_task0(x)
                h_temp = self.linear_input_W_h_task0(h)
                c_temp = self.linear_input_w_c_task0(c_prev)
                i = self.sigmoid_input_task0(x_temp + h_temp)
                return i
            if taskname==1:
                x_temp = self.linear_input_W_x_task1(x)
                h_temp = self.linear_input_W_h_task1(h)
                c_temp = self.linear_input_w_c_task1(c_prev)
                i = self.sigmoid_input_task1(x_temp + h_temp)
                return i
            if taskname==2:
                x_temp = self.linear_input_W_x_task2(x)
                h_temp = self.linear_input_W_h_task2(h)
                c_temp = self.linear_input_w_c_task2(c_prev)
                i = self.sigmoid_input_task2(x_temp + h_temp)
                return i
            if taskname==3:
                x_temp = self.linear_input_W_x_task3(x)
                h_temp = self.linear_input_W_h_task3(h)
                c_temp = self.linear_input_w_c_task3(c_prev)
                i = self.sigmoid_input_task3(x_temp + h_temp)
                return i
    def forget_gate(self, x, h, c_prev,shared,taskname):
        if shared:
            x = self.linear_forget_W_x_shared(x)
            h = self.linear_forget_W_h_shared(h)
            c = self.linear_forget_w_c_shared(c_prev)
            f = self.sigmoid_forget_shared(x + h)
            return f
        else:
            if taskname ==0:
                x = self.linear_forget_W_x_task0(x)
                h = self.linear_forget_W_h_task0(h)
                c = self.linear_forget_w_c_task0(c_prev)
                f = self.sigmoid_forget_task0(x + h)
                return f
            if taskname ==1:
                x = self.linear_forget_W_x_task1(x)
                h = self.linear_forget_W_h_task1(h)
                c = self.linear_forget_w_c_task1(c_prev)
                f = self.sigmoid_forget_task1(x + h)
                return f
            if taskname ==2:
                x = self.linear_forget_W_x_task2(x)
                h = self.linear_forget_W_h_task2(h)
                c = self.linear_forget_w_c_task2(c_prev)
                f = self.sigmoid_forget_task2(x + h)
                return f
            if taskname ==3:
                x = self.linear_forget_W_x_task3(x)
                h = self.linear_forget_W_h_task3(h)
                c = self.linear_forget_w_c_task3(c_prev)
                f = self.sigmoid_forget_task3(x + h)
                return f
    def cell_memory_gate(self, i, f, x, h, c_prev,shared, taskname):
        if shared:
            x = self.linear_cell_W_x_shared(x)
            h = self.linear_cell_W_h_shared(h)
            k = self.tanh_cell_shared(x + h)
            g = (i  * k)
            c = (f * c_prev)
            c_next = g
            return c_next
        else:
            if taskname == 0:
                x = self.linear_cell_W_x_task0(x)
                h = self.linear_cell_W_h_task0(h)
                k = self.tanh_cell_task0(x + h)
                g = (i  * k)
                c = (f * c_prev)
                c_next = g
                return c_next

            if taskname == 1:
                x = self.linear_cell_W_x_task1(x)
                h = self.linear_cell_W_h_task1(h)
                k = self.tanh_cell_task1(x + h)
                g = (i  * k)
                c = (f * c_prev)
                c_next = g
                return c_next
            if taskname == 2:
                x = self.linear_cell_W_x_task2(x)
                h = self.linear_cell_W_h_task2(h)
                k = self.tanh_cell_task2(x + h)
                g = (i * k)
                c = (f * c_prev)
                c_next = g
                return c_next
            if taskname == 3:
                x = self.linear_cell_W_x_task3(x)
                h = self.linear_cell_W_h_task3(h)
                k = self.tanh_cell_task3(x + h)
                g = (i  * k)
                c = (f * c_prev)
                c_next = g
                return c_next
    def out_gate(self, x, h, c_prev, shared, taskname):
        if shared:
            x = self.linear_output_W_x_shared(x)
            # t = self.linear_output_W_t_shared(delt)
            h = self.linear_output_W_h_shared(h)
            c = self.linear_output_w_c_shared(c_prev)
            o = self.sigmoid_out_shared(x + h)
            return o
        else:
            if taskname == 0:
                x = self.linear_output_W_x_task0(x)
                # t = self.linear_output_W_t_task0(delt)
                h = self.linear_output_W_h_task0(h)
                c = self.linear_output_w_c_task0(c_prev)
                o = self.sigmoid_input_task0(x + h )
                return o
            if taskname == 1:
                x = self.linear_output_W_x_task1(x)
                # t = self.linear_output_W_t_task1(delt)
                h = self.linear_output_W_h_task1(h)
                c = self.linear_output_w_c_task1(c_prev)
                o = self.sigmoid_input_task1(x  + h)
                return o
            if taskname == 2:
                x = self.linear_output_W_x_task2(x)
                # t = self.linear_output_W_t_task2(delt)
                h = self.linear_output_W_h_task2(h)
                c = self.linear_output_w_c_task2(c_prev)
                o = self.sigmoid_input_task2(x + h)
                return o
            if taskname == 3:
                x = self.linear_output_W_x_task3(x)
                # t = self.linear_output_W_t_task3(delt)
                h = self.linear_output_W_h_task3(h)
                c = self.linear_output_w_c_task3(c_prev)
                o = self.sigmoid_input_task3(x  + h)
                return o
    def shared_layer(self,x,h_task_specific,tuple_in,shared, taskname):
        (h, c_prev) = tuple_in
        x = torch.cat((x, h_task_specific), axis=2)
        i = self.input_gate(x, h, c_prev,shared,taskname)
        f = self.forget_gate(x, h, c_prev,shared,taskname)
        # T = self.time_gate(x,shared,taskname)
        c_next = self.cell_memory_gate(i, f, x, h, c_prev,shared,taskname)
        o = self.out_gate(x, h, c_prev,shared, taskname)
        h_next = o * self.tanh_cell_shared(c_next)

        return h_next, c_next
    def specific_layer(self,x,h_shared,tuple_in,shared, taskname):
        (h, c_prev) = tuple_in
        input_concat = torch.cat((x, h_shared), axis=2)
        # x = (x,h_shared)
        i = self.input_gate(input_concat, h, c_prev,shared,taskname)
        f = self.forget_gate(input_concat, h, c_prev,shared,taskname)
        # T = self.time_gate(input_concat,shared,taskname)
        c_next = self.cell_memory_gate(i, f, input_concat, h, c_prev,shared,taskname)
        o = self.out_gate(input_concat, h, c_prev,shared,taskname)
        h_next = o * self.tanh_cell_shared(c_next)
        return h_next, c_next
    def forward(self, x, tuple_shared, tuple_in, taskname):
        # x_attention = self.linear_feature_attention_W_x(x)
        # x_attention_sig = self.sigmoid_feature_attention(x_attention)
        # x_attention_out = torch.mul(x_attention, x_attention_sig)
        # print("x_attention_out:", x_attention_out)
        # x = self.multivariate(x)
        (h,c_prev) = tuple_in
        h_next_shared, c_next_shared = self.shared_layer(x,h, tuple_shared, True, taskname)
        h_next_shared_attention = self.linear_shared_layer_attention_W_x(h_next_shared)
        h_next_shared_sig = self.sigmoid_shared_layer_attention(h_next_shared_attention)
        h_next_shared_sig_out = torch.mul(h_next_shared_sig, h_next_shared)
        # print("h_next_shared_sig_out: ", h_next_shared_sig_out)
        h_next, c_next = self.specific_layer(x, h_next_shared_sig_out, tuple_in, False, taskname)
        # h_next2, c_next2 = self.specific_layer(x_attention_sig,h_next_shared_sig, tuple_in2, delt)
        return h_next_shared, c_next_shared, h_next, c_next


class LSTM_single_task(torch.nn.Module):
    def __init__(self, input_length, hidden_length):
        super(LSTM_single_task, self).__init__()

        self.hidden_length = hidden_length
        self.input_length = input_length

        self.multivariate = nn.Linear(self.input_length, self.input_length, bias=True)

        # input gate components
        self.linear_input_W_x = nn.Linear(self.input_length , self.hidden_length, bias=True)
        self.linear_input_W_h = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.linear_input_w_c = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_input = nn.Sigmoid()

        # forget gate components
        self.linear_forget_W_x = nn.Linear(self.input_length , self.hidden_length, bias=True)
        self.linear_forget_W_h = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.linear_forget_w_c = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_forget = nn.Sigmoid()

        self.linear_cell_W_x = nn.Linear(self.input_length , self.hidden_length, bias=True)
        self.linear_cell_W_h = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.tanh_cell = nn.Tanh()

        # out gate components
        self.linear_output_W_x = nn.Linear(self.input_length , self.hidden_length, bias=True)
        self.linear_output_W_h = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.linear_output_w_c = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_out = nn.Sigmoid()
    def input_gate(self, x, h, c_prev):

        x_temp = self.linear_input_W_x(x)
        h_temp = self.linear_input_W_h(h)
        c_temp = self.linear_input_w_c(c_prev)
        i = self.sigmoid_input(x_temp + h_temp)
        return i
    def forget_gate(self, x, h, c_prev):

        x = self.linear_forget_W_x(x)
        h = self.linear_forget_W_h(h)
        c = self.linear_forget_w_c(c_prev)
        f = self.sigmoid_forget(x + h)
        return f
    def cell_memory_gate(self, i, f, x, h, c_prev):

        x = self.linear_cell_W_x(x)
        h = self.linear_cell_W_h(h)
        k = self.tanh_cell(x + h)
        g = (i * k)
        c = (f * c_prev)
        c_next = g
        return c_next
    def out_gate(self, x, h, c_prev):

        x = self.linear_output_W_x(x)
        # t = self.linear_output_W_t_task0(delt)
        h = self.linear_output_W_h(h)
        c = self.linear_output_w_c(c_prev)
        o = self.sigmoid_input(x + h)
        return o
    def forward(self, x, tuple_in):
        (h, c_prev) = tuple_in
        # x = (x,h_shared)
        i = self.input_gate(x, h, c_prev)
        f = self.forget_gate(x, h, c_prev)
        # T = self.time_gate(input_concat,shared,taskname)
        c_next = self.cell_memory_gate(i, f, x, h, c_prev)
        o = self.out_gate(x, h, c_prev)
        h_next = o * self.tanh_cell(c_next)

        return  h_next, c_next

