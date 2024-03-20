import colorama
from colorama import Fore, Back, Style, Cursor

colorama.init(autoreset=True)

layer = int(input("Enter the layer number: "))

all_layers = []
inchannels, outchannels, kernel, stride, padding = 0, 0, 0, 0, 0

width = int(input("Enter the width of the image: "))
height = int(input("Enter the height of the image: "))

print('-'*80)

for i in range(layer):
    if(inchannels or outchannels):
        print('Before Inchannels: ', inchannels)
        print('Before Outchannels: ', outchannels)
    inchannels = int(input("Enter the number of in_channels: "))
    outchannels = int(input("Enter the number of out_channels: "))

    # Conv Layer
    kernelConv = int(input("Enter the kernelConv size: "))
    strideConv = int(input("Enter the strideConv: "))
    padding = int(input("Enter the padding: "))
    outputConvW = ((width - kernelConv + 2 * padding) / strideConv) + 1
    outputConvH = ((height - kernelConv + 2 * padding) / strideConv) + 1
    print(f'Output Conv Layer{i+1}: {outputConvW} * {outputConvH} * {outchannels}')

    # Max Pooling Layer
    kernelMaxPool = int(input("Enter the kernelMaxPool size: "))
    strideMaxPool = int(input("Enter the strideMaxPool: "))
    outputMaxPoolW = ((outputConvW - kernelMaxPool) / strideMaxPool) + 1
    outputMaxPoolH = ((outputConvH - kernelMaxPool) / strideMaxPool) + 1
    print(f'Output Max Pooling Layer{i+1}: {outputMaxPoolW} * {outputMaxPoolH} * {outchannels}')

    customize = f'''
    # Conv Layer {i+1}

    # Input size: {width}*{height}*{inchannels}
    # Spatial extend of each one (kernelConv size), F = {kernelConv}
    # Slide size (strideConv), S = {strideConv}
    # Padding, P = {padding}
    ## Width: (({width} - {kernelConv} + 2 * {padding}) / {strideConv}) + 1 = {outputConvW}
    ## High: (({height} - {kernelConv} + 2 * {padding}) / {strideConv}) + 1 = {outputConvH}
    ## Depth: {outchannels}
    ## Output Conv Layer{i+1}: {outputConvW} * {outputConvH} * {outchannels}

    self.conv{i+1} = torch.nn.Conv2d(in_channels={inchannels}, out_channels={outchannels}, kernel_size={kernelConv}, stride={strideConv}, padding={padding})
    {Back.RED}self.batch{i+1} = torch.nn.BatchNorm2d({outchannels}) #**optional {Back.RESET}
    {Back.RED}self.drop{i+1} = torch.nn.Dropout2d(0.1) #**optional {Back.RESET}
    self.relu{i+1} = torch.nn.ReLU()

    # Max Pooling Layer {i+1}
    # Input size: {outputConvW} * {outputConvH} * {outchannels}
    ## Spatial extend of each one (kernelMaxPool size), F = {kernelMaxPool}
    ## Slide size (strideMaxPool), S = {strideMaxPool}

    # Output Max Pooling Layer {i+1}
    ## Width: (({outputConvW} - {kernelMaxPool}) / {strideMaxPool}) + 1 = {((outputConvW - kernelMaxPool) / strideMaxPool ) + 1}
    ## High: (({outputConvH} - {kernelMaxPool}) / {strideMaxPool}) + 1 = {((outputConvH - kernelMaxPool) / strideMaxPool ) + 1}
    ## Depth: {outchannels}
    ### Output Max Pooling Layer {i+1}: {((outputConvW - kernelMaxPool) / strideMaxPool ) + 1} * {((outputConvH - kernelMaxPool) / strideMaxPool ) + 1} * {outchannels}
    
    self.pool{i+1} = torch.nn.MaxPool2d(kernel_size={kernelMaxPool}, stride={strideMaxPool})'''
    all_layers.append((i+1, customize))

    width = outputMaxPoolW
    height = outputMaxPoolH
    print('\n')

print('-'*80)
for index, string in all_layers:
    print(f"{string}")
print('-'*80)