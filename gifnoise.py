import sys, struct, random

######################################################################
######################### GIF BASIC STRUCTURE# #######################
######################################################################

class GIFHeader:

    def __init__(self, header_bytes):
        self.magic = header_bytes[:0x6]
        # Other header information (packed fields = logical screen descriptor)
        self.width, self.height = struct.unpack('<HH', header_bytes[0x6:0xA])
        self.packed_fields = header_bytes[0xA]
        self.background_color_index = header_bytes[0xB]
        self.pixel_aspect_ratio = header_bytes[0xC]
        self.gct_flag = (self.packed_fields & 0b10000000) >> 7
        self.gct_size = (1 << ((self.packed_fields & 0x7) + 1)) * 3
        
    def to_bytes(self):
        output = bytearray(self.magic)
        output += bytearray(struct.pack('<2H', *[self.width, self.height]))
        output += bytearray([self.packed_fields, self.background_color_index, self.pixel_aspect_ratio])
        return output

class GIFColorTable:

    def __init__(self, color_table_bytes):
        self.table = [(color_table_bytes[x], color_table_bytes[x + 1], color_table_bytes[x + 2]) for x in range(0, len(color_table_bytes), 3)]

    def to_bytes(self):
        flat_table = []
        for rgb in self.table:
            flat_table.extend(rgb)
        return bytearray(flat_table)

class GIFBlock:

    def __init__(self):
        self.size = None

class GIFSubBlock(GIFBlock):

    def __init__(self, sub_block_bytes):
        self.size = len(sub_block_bytes)
        self.value = sub_block_bytes

    def to_bytes(self):
        return bytearray([self.size]) + bytearray(self.value)

class GIFApplicationExtensionBlock(GIFBlock):

    def __init__(self, block_bytes):
        self.block_size = block_bytes[2]
        self.application_id = block_bytes[3:11]
        self.code = block_bytes[11:14]
        self.sub_blocks, i = [], 14
        while block_bytes[i] != 0:
            # block_bytes[i] is the size of the sub block
            self.sub_blocks.append(GIFSubBlock(block_bytes[i + 1:i + 1 + block_bytes[i]]))
            i += 1 + block_bytes[i]
        self.size = i + 1

    def to_bytes(self):
        output = bytearray(b'\x21\xff') + bytearray([self.block_size]) + bytearray(self.application_id) + bytearray(self.code)
        for sub_block in self.sub_blocks:
            output += sub_block.to_bytes()
        output += bytearray(b'\x00')
        return output

class GIFGraphicalExtensionBlock(GIFBlock):

    def __init__(self, block_bytes):
        self.block_size = block_bytes[2]
        self.packed_fields = block_bytes[3]
        self.delay_time = struct.unpack('<H', block_bytes[4:6])[0]
        self.transparent_color_index = block_bytes[6]
        self.block_terminator = block_bytes[7]
        self.size = 8
        # Derived from packed fields
        self.disposal_method = (self.packed_fields & 0b11100) >> 2
        self.user_input = bool((self.packed_fields & 2) >> 1)
        self.is_transparent = bool(self.packed_fields & 1)
        #log(f'Graphical Control Extension: Size={hex(block_size)}, Delay={delay_time}, DisposalMethod={disposal_method}, UserInput={user_input}, isTransparent={is_transparent}, transparentColorIndex={transparent_color_index}, Terminator={hex(block_terminator)}')

    def to_bytes(self):
        output = bytearray(b'\x21\xf9') + bytearray([self.block_size, self.packed_fields])
        output += bytearray(struct.pack('<H', self.delay_time))
        output += bytearray([self.transparent_color_index, self.block_terminator])
        return output

class GIFImageBlockDescriptor(GIFBlock):

    def __init__(self, block_bytes):
        self.left = struct.unpack('<H', block_bytes[0:2])[0]
        self.top = struct.unpack('<H', block_bytes[2:4])[0]
        self.width = struct.unpack('<H', block_bytes[4:6])[0]
        self.height = struct.unpack('<H', block_bytes[6:8])[0]
        self.packed_fields = block_bytes[8]
        self.size = 9
        # Results from packed fields
        self.lct_flag = bool((self.packed_fields & 0x80) >> 7)
        self.interlace_flag = bool((self.packed_fields & 0x40) >> 6)
        self.sort_flag = bool((self.packed_fields & 0x20) >> 5)
        self.reserved = (self.packed_fields & 0x18) >> 3
        self.lct_size = self.packed_fields & 0x7
    def to_bytes(self):
        output = bytearray(struct.pack('<H', self.left)) + bytearray(struct.pack('<H', self.top))
        output += bytearray(struct.pack('<H', self.width)) + bytearray(struct.pack('<H', self.height))
        output += bytearray([self.packed_fields])
        return output

class GIFImageBlock(GIFBlock):

    def __init__(self, block_bytes):
        # Process image descriptor
        self.image_descriptor = GIFImageBlockDescriptor(block_bytes[1:10])
        self.size = 1 + self.image_descriptor.size
        # Process local color table (LCT)
        self.lct = None
        if self.image_descriptor.lct_flag:
            lct_length = 3 * (1 << (self.image_descriptor.lct_size + 1))
            self.lct = GIFColorTable(block_bytes[self.size:self.size + lct_length])
            self.size += lct_length
        # Process Image Data
        self.lzw_min_code_size = block_bytes[self.size]
        self.size += 1
        self.sub_blocks = []
        while block_bytes[self.size] != 0:
            self.sub_blocks.append(GIFSubBlock(block_bytes[self.size + 1 : self.size + 1 + block_bytes[self.size]]))
            self.size += 1 + block_bytes[self.size]
        #print(len(self.sub_blocks), list(map(lambda x: len(x.value), self.sub_blocks)))
        self.size += 1

    def to_bytes(self):
        output = bytearray(b'\x2C') + bytearray(self.image_descriptor.to_bytes())
        if self.lct:
            output += self.lct.to_bytes()
        output += bytearray([self.lzw_min_code_size])
        for sub_block in self.sub_blocks:
            output += sub_block.to_bytes()
        output += bytearray(b'\x00')
        return output

class GIFTerminatorBlock(GIFBlock):

    def __init__(self):
        self.size = 1

    def to_bytes(self):
        return bytearray(b'\x3B')

class GIFImage:

    def __init__(self, filepath):
        f = open(filepath, 'rb').read()
        # Process header
        self.header = GIFHeader(f[:0xD])
        # Process global color table (gct)
        self.gct = None if not self.header.gct_flag else GIFColorTable(f[0xD:0xD + self.header.gct_size])
        # Process blocks
        self.blocks, i = [], 0xD + self.header.gct_size
        while i < len(f):
            block_type = f[i]
            # Extension introducer
            if block_type == 0x21:
                extension_type = f[i + 1]
                # Application Extension label
                if extension_type == 0xFF:
                    block_size = f[i + 2]
                    if block_size == 11: # Application extension (typically)
                        new_block = GIFApplicationExtensionBlock(f[i:])
                        self.blocks.append(new_block)
                        i += new_block.size
                # Graphic Control Extension
                elif extension_type == 0xF9:
                    new_block = GIFGraphicalExtensionBlock(f[i:])
                    self.blocks.append(new_block)
                    i += new_block.size
            # Image
            elif block_type == 0x2C:
                new_block = GIFImageBlock(f[i:])
                self.blocks.append(new_block)
                i += new_block.size
            # Terminator
            elif block_type == 0x3B and i == len(f) - 1:
                self.blocks.append(GIFTerminatorBlock())
                i += 1
            else:
                print(f'Unknown block type: {hex(block_type)}... is GIF corrupt?')
                exit()

    def to_bytes(self):
        output = bytearray(self.header.to_bytes())
        if self.gct is not None:
            output += self.gct.to_bytes()
        for block in self.blocks:
            output += block.to_bytes()
        return output

######################################################################
########################## LZW COMPRESSION ###########################
######################################################################

def process_clear_code(clear_code, end_of_information_code, min_code_size):
        next_code = end_of_information_code + 1
        current_code_size = min_code_size + 1
        # Represents the list of data bytes each code represents
        dictionary = {i: [i] for i in range(clear_code)}
        dictionary[clear_code] = []
        dictionary[end_of_information_code] = None
        previous_code = None
        return next_code, current_code_size, dictionary, previous_code 

def lzw_decompress(compressed_data, min_code_size):
    # Note: Color table has code from 0 to 2^min_code_size - 1
    # --> Each color represented as a code
    # --> additional codes are clear code and end-of-information (EOI) code
    clear_code = 1 << min_code_size
    end_of_information_code = clear_code + 1   

    output = []
    next_code, current_code_size, dictionary, previous_code = process_clear_code(clear_code, end_of_information_code, min_code_size)

    # For each code in compressed data
    bit_position, compressed_data_bitlength = 0, len(compressed_data) << 3
    while bit_position + current_code_size <= compressed_data_bitlength:
        # Compute code for every current_code_size # of bits
        code = 0
        # For every bit of "code" (bit_position to bit_position + current_code_size)
        # Check if bit is 1 or 0 in compressed data to restore original code value
        for i in range(bit_position, bit_position + current_code_size):
            byte_position, bit_offset = i // 8, i % 8
            if compressed_data[byte_position] & (1 << bit_offset):
                code |= 1 << (i - bit_position)
        bit_position += current_code_size

        # Convert code into output
        # Case 1: Code is clear code, reset state of dictionary etc.
        if code == clear_code:
            next_code, current_code_size, dictionary, previous_code = process_clear_code(clear_code, end_of_information_code, min_code_size)
        # Case 2: Code is end-of-information code, stop
        elif code == end_of_information_code:
            break
        # Case 3: Reconstruct original bytes based on code
        else:
            # If the entry for the code does not exist
            entry = dictionary.get(code)
            if entry is None:
                # Case 3a: code represents a new sequence as per compression, add new entry into dictionary
                if code == next_code:
                    entry = dictionary[previous_code] + [dictionary[previous_code][0]]
                # Case 3b: code could not have been generated from compression side, error out
                else:
                    raise ValueError(f"Invalid code: {code}")
            # Add uncompressed data bytes to the output
            output.extend(entry)
            # Based on the previous code
            if previous_code is not None:
                # There will be a new code equivalent to the current code + the first byte of the current entry
                dictionary[next_code] = dictionary[previous_code] + [entry[0]]
                next_code += 1
                # Expand the code table to accomodate new codes as required
                if next_code >= (1 << current_code_size) and current_code_size < 12:
                    current_code_size += 1
            # Track current code as previously seen code
            previous_code = code
    return output

def write_code_buffered(code, buffer, buffer_length, current_code_size, output):
    # Store X-bit code as the X-MSBs of the buffer, buffer length = # of bits stored
    buffer |= code << buffer_length
    buffer_length += current_code_size
    # Flush every 8 bits as a compressed byte in the output
    while buffer_length >= 8:
        output.append(buffer & 0xFF)
        buffer >>= 8
        buffer_length -= 8
    return buffer, buffer_length

def lzw_compress(data, min_code_size):
    clear_code = 1 << min_code_size
    end_of_information_code = clear_code + 1
    next_code = end_of_information_code + 1
    current_code_size = min_code_size + 1
    dictionary = {chr(i): i for i in range(clear_code)}

    # Write clear code as first code in the output
    output, buffer, buffer_length = bytearray(), 0, 0
    buffer, buffer_length = write_code_buffered(clear_code, buffer, buffer_length, current_code_size, output)
    
    current = ""
    for data_byte in data:
        current += chr(data_byte)
        # If the current data byte pattern exist in the dictionary
        if current not in dictionary:
            # Write pattern up till before the current byte as a code in the buffer
            buffer, buffer_length = write_code_buffered(dictionary[current[:-1]], buffer, buffer_length, current_code_size, output)
            # Store current byte pattern as new code in code table if there is space left
            if next_code < (1 << current_code_size):
                dictionary[current] = next_code
                next_code += 1
            # If maximum code table size not reached, store current byte pattern as new code in table and expand code table size
            # up to a certain limit
            elif current_code_size < 12:
                dictionary[current] = next_code
                next_code += 1
                current_code_size += 1
            # Attempt to form new code starting from current data byte
            current = current[-1]           
    # Clear up any remaining unprocessed data bytes that is not codified (but can be) yet
    if current in dictionary:
        buffer, buffer_length = write_code_buffered(dictionary[current], buffer, buffer_length, current_code_size, output)
    # End with end-of-information code
    buffer, buffer_length = write_code_buffered(end_of_information_code, buffer, buffer_length, current_code_size, output)
    # Empty any remaining bits in the buffer
    if buffer_length > 0:
        output.append(buffer & 0xFF)
    return output

######################################################################
############################ STEGANOGRAPHY ###########################
######################################################################

def generate_random_index_list(number_of_indices, seed):
    index_list = list(range(number_of_indices))
    random.Random(seed).shuffle(index_list)
    return index_list

def lsb_encode(data, hidden_data_bits, random_index_list):
    for i in range(len(hidden_data_bits)):
        index = random_index_list[i]
        data[index] = (data[index] & 0xFE) | int(hidden_data_bits[i])

def lsb_decode(encoded_data, stop_pattern, random_index_list):
    hidden_data, hidden_data_bits = bytearray(), []
    for i in range(len(encoded_data)):
        hidden_data_bits.append(str(encoded_data[random_index_list[i]] & 1))
        if len(hidden_data_bits) == 8:
            hidden_data += bytearray([int(''.join(hidden_data_bits), 2)])
            if hidden_data[-len(stop_pattern):] == stop_pattern:
                break
            hidden_data_bits = []
    return hidden_data[:-len(stop_pattern)]

######################################################################
####################### MAIN APPLICATION LOGIC #######################
######################################################################

DATA_END_MARKER = b'SPECIAL_ENDING_MARKER'

def generate_chunks(data, chunk_max_size):
    byte_position, output = 0, []
    while byte_position < len(data):
        chunk_size = len(data) - byte_position
        chunk_size = chunk_max_size if chunk_size > chunk_max_size else chunk_size
        output.append(data[byte_position : byte_position + chunk_size])
        byte_position += chunk_size
    return output

def hide_data_in_image(img, key, data):
    # Obtain list of image blocks in GIF for hiding information + Calculate maximum possible bits to hide
    image_blocks = list(filter(lambda x: type(x) == GIFImageBlock, img.blocks))
    # Find the chunk size for every single image block that is consistent
    xss = map(lambda x: x.value, image_blocks[0].sub_blocks)
    block_chunk_size = len(lzw_decompress([x for xs in xss for x in xs], image_blocks[0].lzw_min_code_size))
    avail_block_chunk_size = block_chunk_size - len(DATA_END_MARKER) * 8
    data_chunk_size = (len(data) // len(image_blocks)) + 1
    # If the entire data to be hidden cannot be stored in chunks, stop immediately
    if data_chunk_size * 8 > avail_block_chunk_size:
        print(f'Cannot store {data_chunk_size * 8} bits per chunk in {avail_block_chunk_size} bits per image block.')
        exit(0)
    else:
        print(f'Using up to {data_chunk_size * 8} of {avail_block_chunk_size} bits per block... ({(data_chunk_size * 8)/avail_block_chunk_size*100}%)')
    data_position = 0
    for i, block in enumerate(image_blocks):
        compressed_frame_data = b''.join(map(lambda x: x.value, block.sub_blocks))
        # Hide as much data as possible (don't need to un/re-compress if all data already hidden)
        recompressed_frame_data = compressed_frame_data
        decompressed_frame_data = lzw_decompress(compressed_frame_data, block.lzw_min_code_size)
        if data_position + data_chunk_size > len(data):
            data_chunk_size = len(data) - data_position
        data_chunk_bits = ''.join(map(lambda x: format(x, '08b'), data[data_position : data_position + data_chunk_size] + bytearray(DATA_END_MARKER)))
        print(f'Hiding bytes {data_position}-{data_position + data_chunk_size} in image block {i} with key "{key + str(i)}" ...')
        data_position += data_chunk_size
        lsb_encode(decompressed_frame_data, data_chunk_bits, generate_random_index_list(block_chunk_size, key + str(i)))
        recompressed_frame_data = lzw_compress(decompressed_frame_data, block.lzw_min_code_size)
        block.sub_blocks = list(map(lambda x: GIFSubBlock(x), generate_chunks(recompressed_frame_data, 255)))
    # Output GIF file with hidden content as a file
    f = open('hidden.gif', 'wb')
    f.write(img.to_bytes())
    f.close()
    

def show_data_in_image(img, key, target_filepath):
    data = bytearray()
    # Obtain list of image blocks in GIF for hiding information
    image_blocks = list(filter(lambda x: type(x) == GIFImageBlock, img.blocks))
    # Find the chunk size for every single image block that is consistent
    xss = map(lambda x: x.value, image_blocks[0].sub_blocks)
    block_chunk_size = len(lzw_decompress([x for xs in xss for x in xs], image_blocks[0].lzw_min_code_size))
    # Enumerate all image blocks
    for i, block in enumerate(image_blocks):
        # decompress data
        compressed_frame_data = b''.join(map(lambda x: x.value, block.sub_blocks))
        decompressed_frame_data = lzw_decompress(compressed_frame_data, block.lzw_min_code_size)
        data += lsb_decode(decompressed_frame_data, DATA_END_MARKER, generate_random_index_list(block_chunk_size, key + str(i)))
        print(f'Recovering bytes in image block {i} with key "{key + str(i)}" ...')
    # Output hidden content to a file
    f = open(target_filepath, 'wb')
    f.write(data)
    f.close()  

# Process CLI arguments
if len(sys.argv) != 5:
    print(f'Steaganopgraphy for GIF89a files')
    print(f'Format: python3 {sys.argv[0]} hide <key> </path/to/image.gif> </path/to/file_to_hide.ext>')
    print(f'Format: python3 {sys.argv[0]} show <key> </path/to/image.gif> </path/to/recovered_file.ext>')
    exit()
option, key, gif_filepath, dest_filepath = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

# Process GIF Image
if option == 'hide':
    hide_data_in_image(GIFImage(gif_filepath), key, open(dest_filepath, 'rb').read())
elif option == 'show':
    show_data_in_image(GIFImage(gif_filepath), key, dest_filepath)
