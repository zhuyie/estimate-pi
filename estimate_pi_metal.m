#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

const uint N_THREADS = 1000*100;
const uint ITERS_PER_THREAD = 10000;

NS_ASSUME_NONNULL_BEGIN

@interface MetalPi : NSObject
- (instancetype) initWithDevice: (id<MTLDevice>) device;
- (void) execute;
@end

NS_ASSUME_NONNULL_END

@implementation MetalPi
{
    id<MTLDevice> _mDevice;

    // The compute pipeline generated from the compute kernel in the .metal shader file.
    id<MTLComputePipelineState> _mPiFunctionPSO;

    // The command queue used to pass commands to the device.
    id<MTLCommandQueue> _mCommandQueue;

    // Buffers to hold data.
    id<MTLBuffer> _mBufferResult;
    
    NSDate* _mStart;
}

- (instancetype) initWithDevice: (id<MTLDevice>) device
{
    self = [super init];
    if (self)
    {
        _mDevice = device;

        NSError* error = nil;
        
        NSString* source = [NSString stringWithContentsOfFile:@"../pi.metal"
            encoding:NSUTF8StringEncoding error:NULL];
        if (source == nil)
        {
            NSLog(@"Failed to load source.");
            return nil;
        }
        
        MTLCompileOptions* compileOptions = [MTLCompileOptions new];
        id<MTLLibrary> lib = [device newLibraryWithSource:source
            options:compileOptions error:&error];
        if (lib == nil)
        {
            NSLog(@"Failed to create library: %@.", error);
            return nil;
        }
/*
        id<MTLLibrary> lib = [_mDevice newDefaultLibrary];
        if (lib == nil)
        {
            NSLog(@"Failed to find the default library.");
            return nil;
        }
*/
        id<MTLFunction> piFunction = [lib newFunctionWithName:@"pi_v1"];
        if (piFunction == nil)
        {
            NSLog(@"Failed to find the pi function.");
            return nil;
        }

        _mPiFunctionPSO = [_mDevice newComputePipelineStateWithFunction: piFunction error:&error];
        if (_mPiFunctionPSO == nil)
        {
            NSLog(@"Failed to create pipeline state object, error %@.", error);
            return nil;
        }

        _mCommandQueue = [_mDevice newCommandQueue];
        if (_mCommandQueue == nil)
        {
            NSLog(@"Failed to create the command queue.");
            return nil;
        }
        
        uint bufferSize = sizeof(uint) * N_THREADS;
        _mBufferResult = [_mDevice newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
        if (_mBufferResult == nil)
        {
            NSLog(@"Failed to create the result buffer.");
            return nil;
        }
    }

    return self;
}

- (void) execute
{
    _mStart = [NSDate date];
    
    // Create a command buffer to hold commands.
    id<MTLCommandBuffer> commandBuffer = [_mCommandQueue commandBuffer];
    assert(commandBuffer != nil);

    // Start a compute pass.
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    assert(computeEncoder != nil);

    [self encodePiCommand:computeEncoder];

    // End the compute pass.
    [computeEncoder endEncoding];

    // Execute the command.
    [commandBuffer commit];

    // Block until the calculation is complete.
    [commandBuffer waitUntilCompleted];
    
    [self printResult];
}

- (void)encodePiCommand:(id<MTLComputeCommandEncoder>)computeEncoder {

    // Encode the pipeline state object and its parameters.
    [computeEncoder setComputePipelineState:_mPiFunctionPSO];
    uint iters = ITERS_PER_THREAD;
    [computeEncoder setBytes:&iters length:sizeof(iters) atIndex:0];
    uint seed = 42;
    [computeEncoder setBytes:&seed length:sizeof(seed) atIndex:1];
    [computeEncoder setBuffer:_mBufferResult offset:0 atIndex:2];

    MTLSize gridSize = MTLSizeMake(N_THREADS, 1, 1);

    // Calculate a threadgroup size.
    NSUInteger groupSize = _mPiFunctionPSO.maxTotalThreadsPerThreadgroup;
    if (groupSize > N_THREADS)
    {
        groupSize = N_THREADS;
    }
    MTLSize threadgroupSize = MTLSizeMake(groupSize, 1, 1);

    // Encode the compute command.
    [computeEncoder dispatchThreads:gridSize
              threadsPerThreadgroup:threadgroupSize];
}

- (void) printResult
{
    unsigned long total = 0;
    uint* result = _mBufferResult.contents;
    for (uint i = 0; i < N_THREADS; i++)
    {
        total += result[i];
    }
    unsigned long long samples = (unsigned long long)ITERS_PER_THREAD * N_THREADS;
    double pi = (double)total / samples * 4;
    double pi_true = acos(-1.0);  // true value of pi
    double error = fabs(pi - pi_true) / pi_true * 100;
    
    NSTimeInterval duration = [_mStart timeIntervalSinceNow];
    
    printf("GPU = %s\n", [_mDevice.name UTF8String]);
    printf("threads = %u\n", N_THREADS);
    printf("iteraters = %u\n", ITERS_PER_THREAD);
    printf("samples = %llu\n", samples);
    printf("pi = %f (%f%% error)\n", pi, error);
    printf("duration = %.2f ms\n", fabs(duration) * 1000);
/*
    CFStringRef msg;
    msg = CFStringCreateWithFormat(
        NULL, NULL,
        CFSTR("GPU = %@\n"
             "threads = %u\n"
             "iteraters = %u\n"
             "samples = %llu\n"
             "pi = %f (%f%% error)\n"
             "duration = %.2f ms"),
        _mDevice.name, N_THREADS, ITERS_PER_THREAD, samples, pi, error, fabs(duration) * 1000);
    CFUserNotificationDisplayNotice(
        0, kCFUserNotificationPlainAlertLevel, NULL, NULL, NULL, CFSTR("estimate_pi_metal"), msg, CFSTR("OK"));
    CFRelease(msg);
*/
}

@end

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        
        MetalPi *pi = [[MetalPi alloc] initWithDevice:device];
        
        [pi execute];
    }
    return 0;
}
