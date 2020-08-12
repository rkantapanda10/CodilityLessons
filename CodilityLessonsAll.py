"""
Lesson 1:Iterations:BinaryGap:Find longest sequence of zeros in binary representation of an integer.
def binary_gap(N):
    return len(max(format(N, 'b').strip('0').split('1')))  
or
def solution(N):
    max_gap = 0
    current_gap = 0
    # Skip the tailing zero(s)
    while N > 0 and N%2 == 0:
        N //= 2
    while N > 0:
        remainder = N%2
        if remainder == 0:
            # Inside a gap
            current_gap += 1
        else:
            # Gap ends
            if current_gap != 0:
                max_gap = max(current_gap, max_gap)
                current_gap = 0
        N //= 2
    return max_gap
Lesson 2:Arrays:CyclicRotation:Rotate an array to the right by a given number of steps.
def solution(A, K):
    # write your code in Python 2.7
    if len(A) == 0:
        return A
    K = K % len(A)
    return A[-K:] + A[:-K]
Lesson 2:Arrays:OddOccurrencesInArray:Find value that occurs in odd number of elements.
def solution(A):
    result = 0
    for number in A:
        result ^= number
    return result
"""


"""
#Lesson 3:Time Complexity:FrogJmp:Count minimal number of jumps from position X to Y.
def solution(X, Y, D):
    if Y < X or D <= 0:
        raise Exception("Invalid arguments")
         
    if (Y- X) % D == 0:
        return (Y- X) // D
    else:
        return ((Y- X) // D) + 1
#Lesson 3:Time Complexity:PermMissingElem:Find the missing element in a given permutation.
        def solution(A):
    should_be = len(A) # you never see N+1 in the iteration
    sum_is = 0
 
    for idx in xrange(len(A)):
        sum_is += A[idx]
        should_be += idx+1
 
    return should_be - sum_is +1

or
def solution(A):
    missing_element = len(A)+1
     
    for idx,value in enumerate(A):
        missing_element = missing_element ^ value ^ (idx+1)
         
    return missing_element
#Lesson 3:Time Complexity:TapeEquilibrium:Minimize the value |(A[0] + ... + A[P-1]) - (A[P] + ... + A[N-1])|.

import sys
 
def solution(A):
    #1st pass
    parts = [0] * len(A)
    parts[0] = A[0]
  
    for idx in xrange(1, len(A)):
        parts[idx] = A[idx] + parts[idx-1]
  
    #2nd pass
    solution = sys.maxint
    for idx in xrange(0, len(parts)-1):
        solution = min(solution,abs(parts[-1] - 2 * parts[idx]));  
  
    return solution
"""


"""
#Lesson 4:Counting Elements:FrogRiverOne:Find the earliest time when a frog can jump to the other side of a river.
def solution(X, A):
    passable = [False] * X
    uncovered = X
 
    for idx in xrange(len(A)):
        if A[idx] <= 0 or A[idx] > X:
            raise Exception("Invalid value", A[idx])
        if passable[A[idx]-1] == False:
            passable[A[idx]-1] = True
            uncovered -= 1
            if uncovered == 0:
                return idx
 
    return -1
#Lesson 4:Counting Elements:MaxCounters:Calculate the values of counters after applying all alternating operations: increase counter by 1; set value of all counters to current maximum.
def solution(N, A):
    result = [0]*N    # The list to be returned
    max_counter = 0   # The used value in previous max_counter command
    current_max = 0   # The current maximum value of any counter
    for command in A:
        if 1 <= command <= N:
            # increase(X) command
            if max_counter > result[command-1]:
                # lazy write
                result[command-1] = max_counter
            result[command-1] += 1
            if current_max < result[command-1]:
                current_max = result[command-1]
        else:
            # max_counter command
            # just record the current maximum value for later write
            max_counter = current_max
    for index in range(0,N):
        if result[index] < max_counter:
            # This element has never been used/updated after previous
            #     max_counter command
            result[index] = max_counter
    return result
#Lesson 4:Counting Elements:MissingInteger:Find the smallest positive integer that does not occur in a given sequence.
def solution(A):
    seen = [False] * len(A)
    for value in A:
        if 0 < value <= len(A):
            seen[value-1] = True
 
    for idx in range(len(seen)):
        if seen[idx] == False:
            return idx + 1
 
    return len(A)+1
#Lesson 4:Counting Elements:PermCheck:Check whether array A is a permutation.
def solution(A):
    seen = [False] * len(A)
 
    for value in A:
        if 0 <= value > len(A):
            return 0
        if seen[value-1] == True:
            return 0
        seen[value-1] = True
 
    return 1

"""
"""
#Lesson 5:Prefix Sums:CountDiv:Compute number of integers divisible by k in range [a..b].
def solution(A, B, K):
    if B < A or K <= 0:
        raise Exception("Invalid Input")
 
    min_value =  ((A + K -1) // K) * K
 
    if min_value > B:
      return 0
 
    return ((B - min_value) // K) + 1
#Lesson 5:Prefix Sums:GenomicRangeQuery:Find the minimal nucleotide from a range of sequence DNA.
def writeCharToList(S, last_seen, c, idx):
    if S[idx] == c:
        last_seen[idx] = idx
    elif idx > 0:
        last_seen[idx] = last_seen[idx -1]
 
def solution(S, P, Q):
     
    if len(P) != len(Q):
        raise Exception("Invalid input")
     
    last_seen_A = [-1] * len(S)
    last_seen_C = [-1] * len(S)
    last_seen_G = [-1] * len(S)
    last_seen_T = [-1] * len(S)
         
    for idx in xrange(len(S)):
        writeCharToList(S, last_seen_A, 'A', idx)
        writeCharToList(S, last_seen_C, 'C', idx)
        writeCharToList(S, last_seen_G, 'G', idx)
        writeCharToList(S, last_seen_T, 'T', idx)
     
     
    solution = [0] * len(Q)
     
    for idx in xrange(len(Q)):
        if last_seen_A[Q[idx]] >= P[idx]:
            solution[idx] = 1
        elif last_seen_C[Q[idx]] >= P[idx]:
            solution[idx] = 2
        elif last_seen_G[Q[idx]] >= P[idx]:
            solution[idx] = 3
        elif last_seen_T[Q[idx]] >= P[idx]:
            solution[idx] = 4
        else:    
            raise Exception("Should never happen")
         
    return solution
#Lesson 5:Prefix Sums:MinAvgTwoSlice:Find the minimal average of any slice containing at least two elements.
def solution(A):
    min_idx = 0
    min_value = 10001
 
    for idx in xrange(0, len(A)-1):
        if (A[idx] + A[idx+1])/2.0 < min_value:
            min_idx = idx
            min_value = (A[idx] + A[idx+1])/2.0
        if idx < len(A)-2 and (A[idx] + A[idx+1] + A[idx+2])/3.0 < min_value:
            min_idx = idx
            min_value = (A[idx] + A[idx+1] + A[idx+2])/3.0
 
    return min_idx
#Lesson 5:Prefix Sums:PassingCars:Count the number of passing cars on the road.
def solution(A):
    west_cars = 0
    cnt_passings = 0

    for idx in xrange(len(A)-1, -1, -1):
        if A[idx] == 0:
            cnt_passings += west_cars
            if cnt_passings > 1000000000:
                return -1
        else:
            west_cars += 1

    return cnt_passings
    
    """
"""
#Lesson 6:Sorting:Distinct:Compute number of distinct values in an array.

def solution(A):
    return len(set(A))

def solution(A):
    if len(A) == 0:
        return 0
 
    A.sort()
 
    nr_values = 1
    last_value = A[0]
 
    for idx in xrange(1, len(A)):
        if A[idx] != last_value:
            nr_values += 1
            last_value = A[idx]
 
    return nr_values

Example test:   [2, 1, 1, 2, 3, 1]
"""

"""#Lesson 6:Sorting: MaxProductOfThree:Maximize A[P] * A[Q] * A[R] for any triplet (P, Q, R).
#https://app.codility.com/c/run/trainingF86NYG-ZM7/
def solution(A):
    if len(A) < 3:
        raise Exception("Invalid input")
         
    A.sort()
     
    return max(A[0] * A[1] * A[-1], A[-1] * A[-2] * A[-3])
#OR
def betterSolution(A):
    if len(A) < 3:
        raise Exception("Invalid input")
         
    minH = []
    maxH = []
     
    for val in A:
        if len(minH) < 2:
            heapq.heappush(minH, -val)
        else:
            heapq.heappushpop(minH, -val)
             
        if len(maxH) < 3:
            heapq.heappush(maxH, val)
        else:
            heapq.heappushpop(maxH, val)
     
     
    max_val = heapq.heappop(maxH) * heapq.heappop(maxH)
    top_ele = heapq.heappop(maxH)
    max_val *= top_ele
    min_val = -heapq.heappop(minH) * -heapq.heappop(minH) * top_ele
     
    return max(max_val, min_val)

print(betterSolution([-3, 1, 2, -2, 5, 6]))"""
"""#Lesson 6:Sorting:NumberOfDiscIntersections:Compute the number of intersections in a sequence of discs





#https://app.codility.com/c/run/trainingTAFBPK-AG8/
def solution(A):
    circle_endpoints = []
    for i, a in enumerate(A):
        circle_endpoints += [(i-a, True), (i+a, False)]
 
    circle_endpoints.sort(key=lambda x: (x[0], not x[1]))
 
    intersections, active_circles = 0, 0
 
    for _, is_beginning in circle_endpoints:
        if is_beginning:
            intersections += active_circles
            active_circles += 1
        else:
            active_circles -= 1
        if intersections > 10E6:
            return -1
 
    return intersections
print(solution([1, 5, 2, 1, 4, 0]))"""

"""#Lesson 6:Sorting:Triangle:Determine whether a triangle can be built from a given set of edges.
#https://app.codility.com/c/run/trainingFE4J9W-APB/
def solution(A):
    if 3 > len(A):
        return 0
 
    A.sort()
 
    for i in range(len(A)-2):
        if A[i] + A[i+1] > A[i+2]:
            return 1
 
    return 0
Example test:   [10, 2, 5, 1, 8, 20]
OK

Example test:   [10, 50, 5, 1]"""
"""
#Lesson 7: Stacks and Queues:Brackets:Determine whether a given string of parentheses (multiple types) is properly nested.
def isValidPair(left, right):
    if left == '(' and right == ')':
        return True
    if left == '[' and right == ']':
        return True 
    if left == '{' and right == '}':
        return True   
    return False
 
def solution(S):
    stack = []
     
    for symbol in S:
        if symbol == '[' or symbol == '{' or symbol == '(':
            stack.append(symbol)
        else:
            if len(stack) == 0:
                return 0
            last = stack.pop()
            if not isValidPair(last, symbol):
                return 0
     
    if len(stack) != 0:
        return 0
             
    return 1
print(solution('{[()()]}'))
print(solution('([)()]'))"""

"""
#Lesson 7: Stacks and Queues:Fish:N voracious fish are moving along a river. Calculate how many fish are alive.
#https://app.codility.com/c/run/trainingMPGEFK-C9G/
def solution(A, B):
    survivals = 0
    stack = []
     
    for idx in range(len(A)):
        if B[idx] == 0:
            while len(stack) != 0:
                if stack[-1] > A[idx]:
                    break
                else:
                    stack.pop()
                         
            else:
                survivals += 1
        else:
            stack.append(A[idx])
             
    survivals += len(stack)
     
    return survivals

print(solution([4, 3, 2, 1, 5], [0, 1, 0, 0, 0]))
"""

"""#Lesson 7: Stacks and Queues:Nesting:Determine whether a given string of parentheses (single type) is properly nested.
def solution(S):
    leftBrackets = 0
     
    for symbol in S:
        if symbol == '(':
            leftBrackets += 1
        else:
            if leftBrackets == 0:
                return 0
            leftBrackets -= 1 
     
    if leftBrackets != 0:
        return 0
     
    return 1

print(solution('(()(())())'))
print(solution('())'))"""
"""#Lesson 7: Stacks and Queues:StoneWall:Cover "Manhattan skyline" using the minimum number of rectangles.
#https://app.codility.com/c/run/trainingYAAXHV-ZMD/
def solution(H):
    block_cnt = 0
     
    stack = []
     
    for height in H:
        # remove all blocks that are bigger than my height
        while len(stack) != 0 and stack[-1] > height:
            stack.pop()
         
        if len(stack) != 0 and stack[-1] == height:
            # we already paid for this size
            pass
        else:
            # new block is required, push it's size to the stack
            block_cnt += 1
            stack.append(height)
             
    return block_cnt

print(solution([8, 8, 5, 7, 9, 8, 7, 4, 8]))
"""
"""
#Lesson 8:Leader: Dominator:Find an index of an array such that its value occurs at more than half of indices in the array.
#https://app.codility.com/c/run/trainingUXUQX8-ABQ/
def solution(A):
    candidate_ele = ''
    candidate_cnt = 0
     
    for value in A:
        if candidate_ele == '':
            candidate_ele = value
            candidate_cnt = 1
        else:
            if value != candidate_ele:
                candidate_cnt -= 1
                if candidate_cnt == 0:
                    candidate_ele = ''
            else:
                candidate_cnt += 1
         
    if candidate_cnt == 0:
        return -1
         
    cnt = 0
    last_idx = 0
     
    for idx, value in enumerate(A):
        if value == candidate_ele:
            cnt += 1
            last_idx = idx
             
    if cnt > len(A)//2:
        return last_idx
         
    return -1
print(solution([3, 4, 3, 2, 3, -1, 3, 3]))
"""



"""
#Lesso 8:Leader:EquiLeader:Find the index S such that the leaders of the sequences A[0], A[1], ..., A[S] and A[S + 1], A[S + 2], ..., A[N - 1] are the same.
#https://app.codility.com/c/run/training3PE35P-U8C/

def solution(A):
    candidate_ele = ''
    candidate_cnt = 0
 
    for value in A:
        if candidate_ele == '':
            candidate_ele = value
            candidate_cnt = 1
        else:
            if value != candidate_ele:
                candidate_cnt -= 1
                if candidate_cnt == 0:
                    candidate_ele = ''
            else:
                candidate_cnt += 1
 
    if candidate_cnt == 0:
        return 0
 
    cnt = 0
    last_idx = 0
 
    for idx, value in enumerate(A):
        if value == candidate_ele:
            cnt += 1
            last_idx = idx
 
    if cnt < len(A)//2:
        return 0
 
    equi_cnt = 0
    cnt_to_the_left = 0
    for idx, value in enumerate(A):
        if value == candidate_ele:
            cnt_to_the_left +=1
        if cnt_to_the_left > (idx + 1)//2 and \
            cnt - cnt_to_the_left > (len(A) - idx - 1) //2:
            equi_cnt += 1
 
    return equi_cnt

print(solution([4, 3, 4, 4, 4, 2]))
"""
"""
#Lesson 9:Maximum slice problem:MaxSliceSum:Find a maximum sum of a compact subsequence of array elements.
#https://app.codility.com/c/run/trainingKR9FTH-WPG/

def solution(A):
    max_ending = max_slice = -1000000
    for a in A:
        max_ending = max(a, max_ending +a)
        max_slice = max(max_slice, max_ending)
         
    return max_slice

print(solution([3, 2, -6, 4, 0]))
"""
"""
#Lesson 9:Maximum slice problem:MaxProfit:Given a log of stock prices compute the maximum possible earning.
#https://app.codility.com/c/run/trainingVT9Z2A-HCA/
def solution(A):
    # write your code in Python 3.6
    max_profit = 0
    max_day = 0
    min_day = 200000
    
    for day in A:
        min_day = min(min_day, day)
        max_profit = max(max_profit, day-min_day)
    
    return max_profit
print(solution([23171, 21011, 21123, 21366, 21013, 21367]))
"""

"""
#Lesson 9:Maximum slice problem:MaxDoubleSliceSum:Find the maximal sum of any double slice.
#https://app.codility.com/c/run/trainingVF2WAF-HH5/
def solution(A):
    ending_here = [0] * len(A)
    starting_here = [0] * len(A)
     
    for idx in range(1, len(A)):
        ending_here[idx] = max(0, ending_here[idx-1] + A[idx])
     
    for idx in reversed(range(len(A)-1)):
        starting_here[idx] = max(0, starting_here[idx+1] + A[idx])
     
    max_double_slice = 0
     
    for idx in range(1, len(A)-1):
        max_double_slice = max(max_double_slice, starting_here[idx+1] + ending_here[idx-1])
         
         
    return max_double_slice

"""
"""
#Lesson 10: Prime and composite numbers:Peaks:Divide an array into the maximum number of same-sized blocks, each of which should contain an index P such that A[P - 1] < A[P] > A[P + 1].
#https://app.codility.com/demo/results/trainingZYBJ9S-TQC/      Score: 27%
def solution(A):
    from math import sqrt
    A_len = len(A)
    next_peak = [-1] * A_len
    peaks_count = 0
    first_peak = -1
    # Generate the information, where the next peak is.
    for index in xrange(A_len-2, 0, -1):
        if A[index] > A[index+1] and A[index] > A[index-1]:
            next_peak[index] = index
            peaks_count += 1
            first_peak = index
        else:
            next_peak[index] = next_peak[index+1]
    if peaks_count < 2:
        # There is no peak or only one.
        return peaks_count
    max_flags = 1
    max_min_distance = int(sqrt(A_len))
    for min_distance in xrange(max_min_distance + 1, 1, -1):
        # Try for every possible distance.
        flags_used = 1
        flags_have = min_distance-1 # Use one flag at the first peak
        pos = first_peak
        while flags_have > 0:
            if pos + min_distance >= A_len-1:
                # Reach or beyond the end of the array
                break
            pos = next_peak[pos+min_distance]
            if pos == -1:
                # No peak available afterward
                break
            flags_used += 1
            flags_have -= 1
        max_flags = max(max_flags, flags_used)
    return max_flags
"""


"""
#Lesson 10: Prime and composite numbers:MinPerimeterRectangle:Find the minimal perimeter of any rectangle whose area equals N.
#https://app.codility.com/c/run/trainingJ4M3K7-4PR/

import math
def solution(N):
    if N <= 0:
      return 0
   
    for i in range(int(math.sqrt(N)), 0, -1):
        if N % i == 0:
            return 2*(i+N//i)
             
    raise Exception("should never reach here!")  

"""

"""
#Lesson 10: Prime and composite numbers:Flags:Find the maximum number of flags that can be set on mountain peaks.
#https://app.codility.com/c/run/trainingXN4W3H-5YT/            score:20%
def solution(A):
    peaks = []
 
    for idx in range(1, len(A)-1):
        if A[idx-1] < A[idx] > A[idx+1]:
            peaks.append(idx)
 
    if len(peaks) == 0:
        return 0
 
    for size in range(len(peaks), 0, -1):
        if len(A) % size == 0:
            block_size = len(A) // size
            found = [False] * size
            found_cnt = 0
            for peak in peaks:
                block_nr = peak//block_size
                if found[block_nr] == False:
                    found[block_nr] = True
                    found_cnt += 1
 
            if found_cnt == size:
                return size
 
    return 0

print(solution([1, 5, 3, 4, 3, 4, 1, 2, 3, 4, 6, 2]))
"""

"""
#Lesson 10: Prime and composite numbers:CountFactors:Count factors of given number n.
#https://app.codility.com/c/run/trainingHUE78C-4X3/
def solution(N):
    cnt = 0
    i = 1
    while ( i * i <= N):
        if (N % i == 0):
            if i * i == N:
               cnt += 1
            else:
                cnt += 2
        i += 1
    return cnt

print(solution(24))
#print(10/2,10%2)
"""


"""
#Lesson 11:Sieve of Eratosthenes:CountNonDivisible:Calculate the number of elements of an array that are not divisors of each element.
#https://app.codility.com/c/run/trainingVT4PQY-JTA/
def solution(A):
  
    A_max = max(A)
  
    count = {}
    for element in A:
        if element not in count:
            count[element] = 1
        else:
            count[element] += 1
  
    divisors = {}
    for element in A:
        divisors[element] = set([1, element])
  
    # start the Sieve of Eratosthenes
    divisor = 2
    while divisor*divisor <= A_max:
        element_candidate = divisor
        while element_candidate  <= A_max:
            if element_candidate in divisors and not divisor in divisors[element_candidate]:
                divisors[element_candidate].add(divisor)
                divisors[element_candidate].add(element_candidate//divisor)
            element_candidate += divisor
        divisor += 1
  
    result = [0] * len(A)
    for idx, element in enumerate(A):
        result[idx] = (len(A)-sum([count.get(divisor,0) for divisor in divisors[element]]))
  
    return result

print(solution([3, 1, 2, 3, 6]))
"""

"""
#Lesson 11:Sieve of Eratosthenes:CountSemiprimes:Count the semiprime numbers in the given range [a..b]
#https://app.codility.com/c/run/trainingY6RNYF-VHU/
def sieve(N):
    semi = set()
    sieve = [True]* (N+1)
    sieve[0] = sieve[1] = False
 
    i = 2
    while (i*i <= N):
        if sieve[i] == True:
            for j in range(i*i, N+1, i):
                sieve[j] = False
        i += 1
 
    i = 2
    while (i*i <= N):
        if sieve[i] == True:
            for j in range(i*i, N+1, i):
                if (j % i == 0 and sieve[j//i] == True):
                    semi.add(j)
        i += 1
 
    return semi
 
def solution(N, P, Q):
 
    semi_set = sieve(N)
 
    prefix = []
 
    prefix.append(0) # 0
    prefix.append(0) # 1
    prefix.append(0) # 2
    prefix.append(0) # 3
    prefix.append(1) # 4
 
    for idx in range(5, max(Q)+1):
        if idx in semi_set:
            prefix.append(prefix[-1]+1)
        else:
            prefix.append(prefix[-1])
 
    solution = []
 
    for idx in range(len(Q)):
        solution.append(prefix[Q[idx]] - prefix[P[idx]-1])
 
    return solution

print(solution(26, [1, 4, 16], [26, 10, 20]))
"""
"""
#Lesson 12: Euclidean algorithm:CommonPrimeDivisors:Check whether two numbers have the same prime divisors.
#https://app.codility.com/c/run/trainingYT49K2-QV4/
def gcd(p, q):
  if q == 0:
    return p
  return gcd(q, p % q)
 
def hasSameFactors(p, q):
    if p == q == 0:
        return True
     
    denom = gcd(p,q)
     
    while (p != 1):
        p_gcd = gcd(p,denom)
        if p_gcd == 1:
            break
        p /= p_gcd
    else:
        while (q != 1):
            q_gcd = gcd(q,denom)
            if q_gcd == 1:
                break
            q /= q_gcd
        else:
            return True
     
    return False
 
 
def solution(A, B):
    if len(A) != len(B):
        raise Exception("Invalid input")
    cnt = 0
    for idx in range(len(A)):
        if A[idx] < 0 or B[idx] < 0:
            raise Exception("Invalid value")
        if hasSameFactors(A[idx], B[idx]):
            cnt += 1
     
    return cnt

print(solution([15, 10, 9], [75, 30, 5]))
"""



"""
#Lesson 12: Euclidean algorithm:ChocolatesByNumbers:There are N chocolates in a circle. Count the number of chocolates you will eat.
#https://app.codility.com/c/run/trainingHS2BMR-3RN/
def gcd(p, q):
  if q == 0:
    return p
  return gcd(q, p % q)
 
def lcm(p,q):
    a= p * (q / gcd(p,q))
    print(p,q,gcd(p,q),q / gcd(p,q),p % q)
    return a
 
def solution(N, M):
    return int(lcm(N,M)/M)

print(solution(10,4))

"""

"""
#Lesson 13:Fibonacci numbers:Ladder:Count the number of different ways of climbing to the top of a ladder.
#https://app.codility.com/c/run/trainingA62P83-QGS/
def solution(A, B):
    # write your code in Python 3.6
    L = max(A)
    P_max = max(B)
    print(L,P_max)
  
    fib = [0] * (L+2)
    fib[1] = 1
    print(fib)
    for i in range(2, L + 2):
        fib[i] = (fib[i-1] + fib[i-2]) & ((1 << P_max) - 1)
        print(fib)
  
    return_arr = [0] * len(A)
    print(return_arr)
    for idx in range(len(A)):
        return_arr[idx] = fib[A[idx]+1] & ((1 << B[idx]) - 1)
        print(return_arr)
  
    return return_arr


print(solution([4, 4, 5, 5, 1], [3, 2, 4, 3, 1]))

"""



"""
#Lesson 13:Fibonacci numbers:FibFrog:Count the minimum number of jumps required for a frog to get to the other side of a river.
def get_fib_seq_up_to_n(N):
    # there are 26 numbers smaller than 100k
    fib = [0] * (27)
    fib[1] = 1
    print(fib)
    for i in range(2, 27):
        fib[i] = fib[i - 1] + fib[i - 2]
        print("a",i,i-1,i-2,fib[i],fib[i-1],fib[i-2],N)
        if fib[i] > N:
            return fib[2:i]
            print(fib)
        else:
            last_valid = i
     
     
     
def solution(A):
    # you can always step on the other shore, this simplifies the algorithm
    A.append(1)
 
    fib_set = get_fib_seq_up_to_n(len(A))
    print(fib_set)
     
    # this array will hold the optimal jump count that reaches this index
    reachable = [-1] * (len(A))
     
    # get the leafs that can be reached from the starting shore
    for jump in fib_set:
        if A[jump-1] == 1:
            reachable[jump-1] = 1
     
    # iterate all the positions until you reach the other shore
    for idx in range(len(A)):
        # ignore non-leafs and already found paths
        if A[idx] == 0 or reachable[idx] > 0:
            continue
 
        # get the optimal jump count to reach this leaf
        min_idx = -1
        min_value = 100000
        for jump in fib_set:
            previous_idx = idx - jump
            if previous_idx < 0:
                break
            if reachable[previous_idx] > 0 and min_value > reachable[previous_idx]:
                min_value = reachable[previous_idx]
                min_idx = previous_idx
        if min_idx != -1:
            reachable[idx] = min_value +1
 
    return reachable[len(A)-1]

print(solution([0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0]))

"""




"""
#Lesson 14:NailingPlanks:Count the minimum number of nails that allow a series of planks to be nailed.
#https://app.codility.com/c/run/trainingBX96CW-W5W/
PLANK_START = 0
PLANK_END = 1
 
NAIL_ARR_IDX = 0
NAIL_HIT_LOCATION = 1
 
class NoNailFoundException(Exception):
    def __init__(self):
        pass
 
def findPosOfNail(nails, plank, previous_max):
    nail_idx = -1
    result = -1
 
    # logarithmic scan O(log(M))
    lower_idx = 0
    upper_idx = len(nails) - 1
 
    while lower_idx <= upper_idx:
        mid_idx = (lower_idx + upper_idx) // 2
        if nails[mid_idx][NAIL_HIT_LOCATION] < plank[PLANK_START]:
            lower_idx = mid_idx + 1
        elif nails[mid_idx][NAIL_HIT_LOCATION] > plank[PLANK_END]:
            upper_idx = mid_idx - 1
        else:
            upper_idx = mid_idx - 1
            result = nails[mid_idx][PLANK_START]
            nail_idx = mid_idx
 
    if result == -1:
        raise NoNailFoundException
 
    # linear scan O(M)
    nail_idx += 1
    while nail_idx < len(nails):
        if nails[nail_idx][NAIL_HIT_LOCATION] > plank[PLANK_END]:
            break
        result = min(result, nails[nail_idx][NAIL_ARR_IDX])
        if result <= previous_max:
            return result
        nail_idx += 1
 
    if result == -1:
        raise NoNailFoundException
 
    return result
 
def getNrNailsRequired(planks, nails):
    result = 0
    for plank in planks:
        result = max(result, findPosOfNail(nails, plank, result))
 
    return result+1
 
def solution(A, B ,C):
    planks = zip(A,B)
 
    nails = sorted(enumerate(C), key=lambda var: var[1])
 
    try:
        return getNrNailsRequired(planks, nails)
    except NoNailFoundException:
        return -1
    
print(solution([1, 4, 5, 8], [4, 5, 9, 10], [4, 6, 7, 10, 2])
"""
"""
#Lesson 14: Binary search algorithm :MinMaxDivision :Divide array A into K blocks and minimize the largest sum of any block.
#https://app.codility.com/c/run/trainingZ3KHNA-Q6H/
def blockSizeIsValid(A, max_block_cnt, max_block_size):
    block_sum = 0
    block_cnt = 0
 
    for element in A:
        if block_sum + element > max_block_size:
            block_sum = element
            block_cnt += 1
        else:
            block_sum += element
        if block_cnt >= max_block_cnt:
            return False
 
    return True
 
def binarySearch(A, max_block_cnt, using_M_will_give_you_wrong_results):
    lower_bound = max(A)
    upper_bound = sum(A)
 
    if max_block_cnt == 1:      return upper_bound
    if max_block_cnt >= len(A): return lower_bound
 
    while lower_bound <= upper_bound:
        candidate_mid = (lower_bound + upper_bound) // 2
        if blockSizeIsValid(A, max_block_cnt, candidate_mid):
            upper_bound = candidate_mid - 1
        else:
            lower_bound = candidate_mid + 1
 
    return lower_bound
 
def solution(K, M, A):
    return binarySearch(A,K,M)

print(solution(3, 5, [2, 1, 5, 1, 2, 2, 2]))"""
"""#lesson 15: Caterpillar:MinAbsSumOfTwo:Find the minimal absolute value of a sum of two elements.
def solution(A):
    N = len(A)
    if N == 1:
        return abs(A[0] * 2)

    A.sort()
    print(A)
    front = 0
    back = N - 1
    min_sum = abs(A[front] + A[back])
    
    for fid in range(0, len(A)):
        for bid in range(0,len(A)):
            if abs(A[fid]+A[bid])< min_sum:
                min_sum=abs(A[fid]+A[bid])
            print("c",fid,bid,A[fid],A[bid],min_sum)  

or
      while front < back:        
        new_front = front + 1
        new_back = back - 1
        print(min_sum,front,back,new_front,new_back)
        
        new_val_front = abs(A[new_front] + A[back])
        new_val_back = abs(A[front] + A[new_back])
        print("a",new_val_front,A[new_front],A[back],new_val_back,A[front],A[new_back])

        if new_val_front < new_val_back:
            min_sum = min(new_val_front, min_sum)
            front = new_front
            print("b",front,min_sum)
        else:
            min_sum = min(new_val_back, min_sum)
            back = new_back
            print("c",front,min_sum)

    return min_sum

print(solution([-8, 4, 5, -10, 3]))"""

"""#lesson 15: Caterpillar: CountTriangles:Count the number of triangles that can be built from a given set of edges.
def solution(A):
    N = len(A)

    if N < 3:
        return 0

    A.sort()
    num_triangles = 0

    for left in range(0, N - 2):
        right = left + 2
        print("a",right,left,A)

        for mid in range(left + 1, N - 1):
            print("-b",left+1,N-1)
            while right < N and A[left] + A[mid] > A[right]:
                num_triangles += right - mid
                print("--c",num_triangles,right,mid, N,A[left], A[mid] , A[right])
                print("")
                right += 1                

    return num_triangles

print(solution([10, 2, 5, 1, 8, 12]))
"""
"""#lesson 15: Caterpillar: CountDistinctSlices:Count the number of distinct slices (containing only unique numbers).
MAX_SLICES = 1000000000
def solution(M, A):
    # NOT ABLE TO
    N=len(A)
    a=b=count=countb=0
    check={}
    while a < N:
        while b < N and A[b] not in check:
            countb += 1
            check[A[b]] = True
            b += 1
            print("a",a,b,countb)

        check.pop(A[a], None)
        print(a,check)

        a += 1
        count += countb
        countb -= 1

        if count >= MAX_SLICES:
            return MAX_SLICES

    return count

print(solution(6,[3,4,5,5,2]))
"""

"""#Lesson 15:Caterpillar method: AbsDistinct:Compute number of distinct absolute values of sorted array elements
def ab():
    check = {}
    count = 0
    A=[-5, -3, -1, 0, 3, 6]

    for n in A:
        key = abs(n)
        print("a",key)
        if key not in check:
            count += 1
            check[key] = True
            print("b",key,check[key],coujn)

    return count

print(ab())
"""





"""#Lesson 16:Greedy algorithm: Tieropes: Tie adjacent ropes to achieve the maximum number of ropes of length >= K.
def fun():
    K=4
    A=[1, 2, 3, 4, 1, 1, 3]
    N=len(A)
    ropes=0
    #left = 0
    right = 0
    temp_sum = 0

    while right < N:
        temp_sum += A[right]
        print("1",temp_sum,A[right],right)

        if temp_sum >= K:
            ropes += 1
            #left = right + 1
            right+=1 #left
            temp_sum = 0
            print("2",temp_sum,right,ropes)
        else:
            right += 1
            print("3",temp_sum,right,ropes)

    return ropes 

print(fun())




#Lesson 16:Greedy algorithm: MaxNonoverlappingSegments : Find a maximal set of non-overlapping segments.   
A=[1, 3, 7, 9, 9]
B=[5, 6, 8, 9, 10]
       
cnt=1
prv_end=B[0]
print("c1:{0},prv_end1:{1}",cnt,prv_end)    
for id in range(1, len(A)):
  if(A[id]>prv_end):
     print("c2:{0},prv_end2:{1},id2:{3}",cnt,prv_end,id)
     cnt+=1
     prv_end=B[id]
     print("c:{0},prv_end:{1},id:{3}",cnt,prv_end,id)

print("cnt:",cnt)"""


#not required:

#https://www.techiedelight.com/find-smallest-missing-element-sorted-array/
#Demo
"""if __name__ == '__main__':
    A = [0, 1, 3, 5, 6]
    m = max(A)
    possible = set(e for e in range(1, m + 2)) - set(A)
    print(possible)

#1st covering prefix
#Demo
def solution():
    A=[2, 2, 1, 0, 1]
    input_set = set(A)
    #print(input_set)
    counter = 0
 
    while input_set:
        if A[counter] in input_set:
            input_set.remove(A[counter])
            print("M",A,counter)
            print("N",input_set)
 
        counter += 1
        print(counter)
 
    return counter-1

print(solution())
"""
  