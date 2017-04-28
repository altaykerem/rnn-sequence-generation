function minibatch(data, batchsize;atype=Array{Float32})
	batch_N = div(length(data),batchsize)
	sequence = [ zeros(3, batchsize) for i=1:batch_N ]

	idx = 0
    for p in data
		ibatch = 1 + idx % batch_N 
		ind = 1+div(idx, batch_N)
		ind > batchsize && break
		sequence[ibatch][:,ind] = p
		idx += 1
    end
    return map(d->convert(atype,d), sequence)
end

function loaddata(file; path="C:\\Users\\HP\\Desktop\\Comp 441\\Sequence Generation project\\$file")
	isfile(path)
	users = readdir(path)
	strokes = Any[]
	### get the input from file
	info("Reading data...")
	for form in users
		linedirs = readdir("$path\\$form")
		for linedir in linedirs
			line = readdir("$path\\$form\\$linedir")
			#push!(lines, line)
			for xml in line
				doc = parse_file("$path\\$form\\$linedir\\$xml")
				docroot = root(doc) # whiteboard session
				strokeset = find_element(docroot, "StrokeSet")
				for stroke in child_elements(strokeset)
					prevx, prevy = 0, 0
					for point in child_elements(stroke)
						x = parse(Int, attribute(XMLElement(point), "x"))
						y = parse(Int, attribute(XMLElement(point), "y"))
						push!(strokes, [x-prevx,y-prevy,0])		#keep x,y offsets from previous input
						prevx = x
						prevy = y
					end
					strokes[end][3] = 1							#1 indicating end of stroke
					free(stroke)
				end
			end
			break
		end
	end
	return strokes
end