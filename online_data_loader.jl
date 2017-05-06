Pkg.installed("LightXML") == nothing && Pkg.add("LightXML")
Pkg.installed("Compat") == nothing && Pkg.add("Compat")
using LightXML
using Compat
include("utils.jl")

function main()
	s, a = synthesisData()
	plotStrokes(s[1]; title = a[1])
end

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

function minibatch_synth(strokes, lines, batchsize;atype=Array{Float32})
	batch_N = div(length(strokes),batchsize)
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

function synthesisData()
	path = "C:\\Users\\HP\\Desktop\\Comp 441\\Sequence Generation project\\"
	strokeDir = "$path\\lineStrokes"
	asciiDir = "$path\\ascii"
	strokes = Any[]
	asciis = Any[]

	info("Reading data...")
	for form in readdir(strokeDir) 	#a01
		linedirs = readdir("$strokeDir\\$form")
		for linedir in linedirs #a01-000
			line = readdir("$strokeDir\\$form\\$linedir")
			asciiu = readdir("$asciiDir\\$form\\$linedir")
			
			#read asciis
			txtfile = asciiu[1] #a01-000u.txt
			text = map((@compat readstring), Any["$asciiDir\\$form\\$linedir\\$txtfile"])[1]
			text = text[match(r"CSR:", text).offset+4:end] #find regex csr:
			lines = split(text,r"\r\n|\n| \r\n";keep=false)
			
			#check if lines match, missmacth due to extra xml files for the same line happens
			if length(lines)!= length(line); break; end
			
			#read strokes
			currentLine = 1
			for xml in line #a01-000u-01.xml
				stroke = Any[]
				doc = parse_file("$strokeDir\\$form\\$linedir\\$xml")
				readOnlineXml(doc, stroke)
				
				###createdata
				push!(strokes, stroke)
				push!(asciis, lines[currentLine])
				currentLine += 1
			end
			break
		end
	end
	info("Read, ",length(asciis)," lines")
	return strokes, asciis
end

function readOnlineXml(doc, strokes)
	docroot = root(doc)
	
	###get minimum x and y values
	v_offsets = find_element(find_element(docroot, "WhiteboardDescription"), "VerticallyOppositeCoords")
	h_offsets = find_element(find_element(docroot, "WhiteboardDescription"), "HorizontallyOppositeCoords")
	x_min = parse(Int, attribute(XMLElement(v_offsets), "x"))
	y_min = parse(Int, attribute(XMLElement(h_offsets), "y"))

	###read all points
	strokeset = find_element(docroot, "StrokeSet")
	for stroke in child_elements(strokeset)
		prevx, prevy = 0, 0
		for point in child_elements(stroke)
			x = parse(Int, attribute(XMLElement(point), "x")) - x_min
			y = parse(Int, attribute(XMLElement(point), "y")) - y_min
			push!(strokes, [x-prevx,y-prevy,0])		#keep x,y offsets from previous input
			prevx = x
			prevy = y
		end
		strokes[end][3] = 1							#1 indicating end of stroke
		free(stroke)
	end
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

main()