#include "StepCsgParser.h"
#include <charconv>
#include <cctype>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

namespace csg {
namespace {
enum class Kind { Identifier, Reference, Number, String, Enum, LParen, RParen, Comma, Equal, Semi, Omitted, Derived, End };
struct Token { Kind kind; std::string text; std::size_t offset; };
std::string upper(std::string s) {
    for (char& c:s) c=static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return s;
}
class Lexer {
public:
    explicit Lexer(std::string_view s): source(s) {
        if (source.size()>=3 && source.substr(0,3)=="\xEF\xBB\xBF") pos=3;
    }
    Token next() {
        for (;;) {
            while(pos<source.size() && std::isspace(static_cast<unsigned char>(source[pos]))) ++pos;
            if(source.substr(pos,2)!="/*") break;
            const auto end=source.find("*/",pos+2);
            if(end==std::string_view::npos) fail("Unterminated comment");
            pos=end+2;
        }
        const auto start=pos;
        if(pos==source.size()) return {Kind::End,"",pos};
        const char c=source[pos++];
        switch(c) {
        case '(': return {Kind::LParen,"(",start};
        case ')': return {Kind::RParen,")",start};
        case ',': return {Kind::Comma,",",start};
        case '=': return {Kind::Equal,"=",start};
        case ';': return {Kind::Semi,";",start};
        case '$': return {Kind::Omitted,"$",start};
        case '*': return {Kind::Derived,"*",start};
        case '#': {
            while(pos<source.size() && std::isdigit(static_cast<unsigned char>(source[pos]))) ++pos;
            if(pos==start+1) fail("Missing entity reference number");
            return {Kind::Reference,std::string(source.substr(start+1,pos-start-1)),start};
        }
        case '\'': {
            std::string value;
            while(pos<source.size()) {
                const char d=source[pos++];
                if(d=='\'') {
                    if(pos<source.size() && source[pos]=='\'') { ++pos; value+='\''; }
                    else return {Kind::String,value,start};
                } else value+=d;
            }
            fail("Unterminated string");
        }
        default: break;
        }
        if(c=='.' && pos<source.size() && std::isalpha(static_cast<unsigned char>(source[pos]))) {
            while(pos<source.size() && (std::isalnum(static_cast<unsigned char>(source[pos])) || source[pos]=='_')) ++pos;
            if(pos==source.size() || source[pos++]!='.') fail("Unterminated enumeration");
            return {Kind::Enum,upper(std::string(source.substr(start+1,pos-start-2))),start};
        }
        if(std::isdigit(static_cast<unsigned char>(c)) || c=='+' || c=='-' || c=='.') {
            while(pos<source.size()) {
                const char d=source[pos];
                if(!(std::isdigit(static_cast<unsigned char>(d)) || d=='.' || d=='e' || d=='E' || d=='d' || d=='D' || d=='+' || d=='-')) break;
                ++pos;
            }
            return {Kind::Number,std::string(source.substr(start,pos-start)),start};
        }
        if(std::isalpha(static_cast<unsigned char>(c)) || c=='_') {
            while(pos<source.size()) {
                const unsigned char d=static_cast<unsigned char>(source[pos]);
                if(!(std::isalnum(d) || d=='_' || d=='-')) break;
                ++pos;
            }
            return {Kind::Identifier,upper(std::string(source.substr(start,pos-start))),start};
        }
        fail("Unsupported STEP character");
    }
private:
    [[noreturn]] void fail(const char* message) const { throw std::runtime_error(std::string(message)+" at byte "+std::to_string(pos)); }
    std::string_view source;
    std::size_t pos=0;
};
enum class ValueKind { Reference, Number, String, Enum, List, Typed, Omitted, Derived };
struct Value {
    ValueKind kind=ValueKind::Omitted;
    std::uint64_t ref=0;
    double number=0;
    std::string text;
    std::vector<Value> children;
};
struct Entity { std::uint64_t id=0; std::string type; std::vector<Value> args; };
using Table=std::map<std::uint64_t,Entity>;
class Reader {
public:
    explicit Reader(std::string_view text): lexer(text), current(lexer.next()) {}
    Table read() {
        identifier("ISO-10303-21"); consume(Kind::Semi);
        identifier("HEADER"); consume(Kind::Semi);
        while(!isIdentifier("ENDSEC")) skipStatement();
        advance(); consume(Kind::Semi);
        identifier("DATA");
        if(current.kind==Kind::LParen) (void)list(0); // optional named DATA section
        consume(Kind::Semi);
        Table table;
        while(!isIdentifier("ENDSEC")) {
            const auto id=referenceToken(); consume(Kind::Equal);
            Entity entity; entity.id=id;
            if(current.kind==Kind::LParen) {
                // Complex units/representation contexts are syntactically read
                // but intentionally not assigned CSG semantics.
                entity.type="<COMPLEX>"; advance();
                while(current.kind!=Kind::RParen) {
                    if(current.kind!=Kind::Identifier) fail("Expected complex entity component");
                    Value part; part.kind=ValueKind::Typed; part.text=current.text; advance();
                    part.children=list(0); entity.args.push_back(std::move(part));
                }
                consume(Kind::RParen);
            } else {
                if(current.kind!=Kind::Identifier) fail("Expected entity name");
                entity.type=current.text; advance(); entity.args=list(0);
            }
            consume(Kind::Semi);
            if(!table.emplace(id,std::move(entity)).second) fail("Duplicate entity ID #"+std::to_string(id));
        }
        advance(); consume(Kind::Semi);
        identifier("END-ISO-10303-21"); consume(Kind::Semi); consume(Kind::End);
        return table;
    }
private:
    void advance() { current=lexer.next(); }
    bool isIdentifier(std::string_view name) const { return current.kind==Kind::Identifier && current.text==name; }
    [[noreturn]] void fail(const std::string& s) const { throw std::runtime_error(s+" at byte "+std::to_string(current.offset)); }
    void consume(Kind kind) { if(current.kind!=kind) fail("Unexpected token '"+current.text+"'"); advance(); }
    void identifier(std::string_view name) { if(!isIdentifier(name)) fail("Expected "+std::string(name)); advance(); }
    std::uint64_t referenceToken() {
        if(current.kind!=Kind::Reference) fail("Expected #reference");
        std::uint64_t id=0;
        const auto result=std::from_chars(current.text.data(),current.text.data()+current.text.size(),id);
        if(result.ec!=std::errc{} || id==0) fail("Invalid entity ID");
        advance(); return id;
    }
    void skipStatement() {
        while(current.kind!=Kind::Semi) {
            if(current.kind==Kind::End) fail("Unterminated header statement");
            advance();
        }
        advance();
    }
    std::vector<Value> list(unsigned depth) {
        if(depth>128) fail("STEP argument nesting exceeds 128");
        consume(Kind::LParen); std::vector<Value> values;
        if(current.kind!=Kind::RParen) {
            for(;;) {
                values.push_back(value(depth+1));
                if(current.kind!=Kind::Comma) break;
                advance();
            }
        }
        consume(Kind::RParen); return values;
    }
    Value value(unsigned depth) {
        Value v;
        switch(current.kind) {
        case Kind::Reference: v.kind=ValueKind::Reference; v.ref=referenceToken(); return v;
        case Kind::Number: {
            v.kind=ValueKind::Number;
            std::string n=current.text;
            for(char& c:n) if(c=='d' || c=='D') c='E';
            const char* begin=n.data(); if(!n.empty() && n[0]=='+') ++begin;
            auto result=std::from_chars(begin,n.data()+n.size(),v.number,std::chars_format::general);
            if(result.ec!=std::errc{} || result.ptr!=n.data()+n.size() || !std::isfinite(v.number)) fail("Invalid finite STEP number");
            advance(); return v;
        }
        case Kind::String: v.kind=ValueKind::String; v.text=current.text; advance(); return v;
        case Kind::Enum: v.kind=ValueKind::Enum; v.text=current.text; advance(); return v;
        case Kind::LParen: v.kind=ValueKind::List; v.children=list(depth); return v;
        case Kind::Identifier: v.kind=ValueKind::Typed; v.text=current.text; advance(); v.children=list(depth); return v;
        case Kind::Omitted: v.kind=ValueKind::Omitted; advance(); return v;
        case Kind::Derived: v.kind=ValueKind::Derived; advance(); return v;
        default: fail("Expected STEP parameter");
        }
    }
    Lexer lexer;
    Token current;
};
[[noreturn]] void bad(const Entity& e, const std::string& why) {
    throw std::runtime_error("STEP #"+std::to_string(e.id)+" "+e.type+": "+why);
}
const Value& arg(const Entity& e,std::size_t i,ValueKind kind) {
    if(i>=e.args.size() || e.args[i].kind!=kind) bad(e,"Invalid argument "+std::to_string(i+1));
    return e.args[i];
}
void arity(const Entity& e, std::size_t n) { if(e.args.size()!=n) bad(e,"Expected "+std::to_string(n)+" arguments"); }
class Resolver {
public:
    explicit Resolver(const Table& t): table(t) {}
    CsgModel resolve() {
        std::set<std::uint64_t> solids;
        for(const auto& [id,e]:table) if(e.type=="CSG_SHAPE_REPRESENTATION") {
            arity(e,3);
            for(const auto& item:arg(e,1,ValueKind::List).children) {
                if(item.kind!=ValueKind::Reference) bad(e,"Representation item must be a reference");
                const auto& target=get(item.ref);
                if(target.type=="CSG_SOLID") {
                    solids.insert(target.id);
                    if(model.representationId==0) model.representationId=id;
                }
            }
        }
        if(solids.empty()) throw std::runtime_error("No CSG_SHAPE_REPRESENTATION -> CSG_SOLID root found");
        if(solids.size()!=1) throw std::runtime_error("Multiple distinct represented CSG solids; split the file into one solid per render");
        model.solidId=*solids.begin(); const auto& solid=get(model.solidId); arity(solid,2);
        model.rootExpressionId=arg(solid,1,ValueKind::Reference).ref;
        model.root=expression(model.rootExpressionId,0);
        return std::move(model);
    }
private:
    const Entity& get(std::uint64_t id) const {
        const auto it=table.find(id);
        if(it==table.end()) throw std::runtime_error("Unresolved STEP reference #"+std::to_string(id));
        return it->second;
    }
    Vec3 vector(std::uint64_t id,const char* type) const {
        const auto& e=get(id); if(e.type!=type) bad(e,"Expected "+std::string(type)); arity(e,2);
        const auto& a=arg(e,1,ValueKind::List).children;
        if(a.size()!=3) bad(e,"Expected three coordinates");
        for(const auto& v:a) if(v.kind!=ValueKind::Number) bad(e,"Non-numeric coordinate");
        return {a[0].number,a[1].number,a[2].number};
    }
    Placement placement(std::uint64_t id) const {
        const auto& e=get(id);
        if(e.type!="AXIS2_PLACEMENT_3D") bad(e,"Expected AXIS2_PLACEMENT_3D"); arity(e,4);
        Placement p; p.origin=vector(arg(e,1,ValueKind::Reference).ref,"CARTESIAN_POINT");
        if(e.args[2].kind!=ValueKind::Omitted) p.z=normalized(vector(arg(e,2,ValueKind::Reference).ref,"DIRECTION"));
        if(e.args[3].kind!=ValueKind::Omitted) p.x=normalized(vector(arg(e,3,ValueKind::Reference).ref,"DIRECTION"));
        else if(std::abs(p.z.x)>1.0-1e-12) p.x={0,1,0};
        const auto projected=p.x-p.z*dot(p.x,p.z);
        if(length(projected)<1e-12) bad(e,"axis and ref_direction are parallel");
        p.x=normalized(projected); p.y=normalized(cross(p.z,p.x)); p.x=normalized(cross(p.y,p.z));
        return p;
    }
    double positive(const Entity& e,std::size_t i) const {
        const double n=arg(e,i,ValueKind::Number).number;
        if(!(n>0)) bad(e,"Dimensions/radius must be positive"); return n;
    }
    Primitive primitive(const Entity& e) const {
        Primitive p; p.stepId=e.id;
        if(e.type=="BLOCK") {
            arity(e,5); p.type=PrimitiveType::Box;
            p.placement=placement(arg(e,1,ValueKind::Reference).ref);
            p.size={positive(e,2),positive(e,3),positive(e,4)};
        } else {
            arity(e,3); p.type=PrimitiveType::Sphere;
            // Standard STEP is SPHERE(name,radius,centre:point), as supplied.
            // Also accept the requested placement form in either argument order.
            std::uint64_t position=0;
            if(e.args[1].kind==ValueKind::Number) { p.radius=positive(e,1); position=arg(e,2,ValueKind::Reference).ref; }
            else { position=arg(e,1,ValueKind::Reference).ref; p.radius=positive(e,2); }
            if(get(position).type=="CARTESIAN_POINT") p.placement.origin=vector(position,"CARTESIAN_POINT");
            else p.placement=placement(position);
        }
        return p;
    }
    std::uint32_t expression(std::uint64_t id,unsigned depth) {
        if(depth>=MaxNodes) throw std::runtime_error("CSG depth exceeds capacity");
        if(auto it=resolved.find(id);it!=resolved.end()) return it->second;
        if(!active.insert(id).second) throw std::runtime_error("Cyclic CSG reference #"+std::to_string(id));
        const auto& e=get(id); CsgNode n; n.stepId=id;
        if(e.type=="BOOLEAN_RESULT") {
            arity(e,4); const auto& op=arg(e,1,ValueKind::Enum).text;
            if(op=="UNION") n.gpu.type=NodeType::Union;
            else if(op=="INTERSECTION") n.gpu.type=NodeType::Intersection;
            else if(op=="DIFFERENCE") n.gpu.type=NodeType::Difference;
            else bad(e,"Unsupported Boolean operator "+op);
            n.gpu.left=expression(arg(e,2,ValueKind::Reference).ref,depth+1);
            n.gpu.right=expression(arg(e,3,ValueKind::Reference).ref,depth+1);
        } else if(e.type=="BLOCK" || e.type=="SPHERE") {
            if(model.operands.size()>=MaxOperands) bad(e,"Operand capacity exceeded");
            n.gpu.type=NodeType::Leaf; n.gpu.operandId=static_cast<std::uint32_t>(model.operands.size());
            model.operands.push_back(primitive(e));
        } else bad(e,"Unsupported entity reachable from CSG tree");
        if(model.nodes.size()>=MaxNodes) bad(e,"Node capacity exceeded");
        const auto index=static_cast<std::uint32_t>(model.nodes.size());
        model.nodes.push_back(n); active.erase(id); resolved.emplace(id,index); return index;
    }
    const Table& table;
    CsgModel model;
    std::set<std::uint64_t> active;
    std::unordered_map<std::uint64_t,std::uint32_t> resolved;
};
}
CsgModel parseStepCsg(std::string_view text) {
    const auto table=Reader(text).read(); return Resolver(table).resolve();
}
CsgModel loadStepCsg(const std::filesystem::path& path) {
    std::ifstream file(path,std::ios::binary);
    if(!file) throw std::runtime_error("Cannot open STEP file: "+path.string());
    const std::string text((std::istreambuf_iterator<char>(file)),std::istreambuf_iterator<char>());
    if(file.bad()) throw std::runtime_error("Failed reading STEP file: "+path.string());
    return parseStepCsg(text);
}
void printTree(const CsgModel& m,std::ostream& out) {
    out << "CSG root: representation #" << m.representationId << " -> solid #" << m.solidId << " -> expression #" << m.rootExpressionId << '\n';
    std::function<void(std::uint32_t,unsigned)> print=[&](std::uint32_t index,unsigned indent) {
        const auto& n=m.nodes.at(index); const std::string pad(indent,' ');
        if(n.gpu.type!=NodeType::Leaf) {
            out << pad << nodeName(n.gpu.type) << " (#" << n.stepId << ")\n";
            print(n.gpu.left,indent+2); print(n.gpu.right,indent+2);
        } else {
            const auto& p=m.operands.at(n.gpu.operandId);
            out << pad << primitiveName(p.type) << " (#" << p.stepId << ", operand " << n.gpu.operandId << ")\n";
            if(p.type==PrimitiveType::Box) out << pad << "  size = " << p.size.x << " x " << p.size.y << " x " << p.size.z << '\n';
            else out << pad << "  radius = " << p.radius << '\n';
            out << pad << "  origin = " << p.placement.origin << "\n" << pad << "  X = " << p.placement.x << ", Y = " << p.placement.y << ", Z = " << p.placement.z << '\n';
        }
    };
    print(m.root,0);
    out << "CSG: nodes = " << m.nodes.size() << ", operands = " << m.operands.size() << ", root = " << nodeName(m.nodes.at(m.root).gpu.type) << '\n';
}
}
